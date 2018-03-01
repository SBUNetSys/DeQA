#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run predictions using the full DrQA retriever-reader pipeline."""

import argparse
import json
import logging
import os
import regex
import sys
import time
from collections import Counter
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from drqa import tokenizers
from drqa.retriever import LuceneRanker
from drqa.reader.utils import slugify

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s.%(msecs)03d: [ %(message)s ]', '%m/%d/%Y_%H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

PROCESS_TOK = None
PROCESS_CANDS = None


def init(tokenizer_class, tokenizer_opts, candidates=None):
    global PROCESS_TOK, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_CANDS = candidates


def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def _split_doc(doc):
    """Given a doc, split it into chunks (by paragraph)."""
    curr = []
    curr_len = 0
    for split in regex.split(r'\n+', doc):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split) > 0:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        yield ' '.join(curr)


def process_batch(retriever, questions, save_dir, n_docs=5, processes=None, num_workers=os.cpu_count()):
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        raise ValueError('save dir not specified!')
    t3 = time.time()
    logger.info('Processing %d queries...' % len(questions))
    logger.info('Retrieving top %d docs...' % n_docs)
    # Rank documents for queries.
    if len(questions) == 1:
        ranked = [retriever.closest_docs(questions[0], k=n_docs)]
    else:
        ranked = retriever.batch_closest_docs(questions, k=n_docs, num_workers=num_workers)

    t4 = time.time()
    logger.info('docs retrieved [time]: %.4f s' % (t4 - t3))
    all_doc_ids, all_doc_scores, all_doc_texts = zip(*ranked)
    # Flatten document ids and retrieve text from database.
    # We remove duplicates for processing efficiency.
    flat_docids, flat_doc_texts = zip(*{(d, t) for doc_ids, doc_texts in zip(all_doc_ids, all_doc_texts)
                                        for d, t in zip(doc_ids, doc_texts)})

    did2didx = {did: didx for didx, did in enumerate(flat_docids)}

    # Split and flatten documents. Maintain a mapping from doc (index in
    # flat list) to split (index in flat list).
    flat_splits = []
    didx2sidx = []
    for text in flat_doc_texts:
        splits = _split_doc(text)
        didx2sidx.append([len(flat_splits), -1])
        for split in splits:
            flat_splits.append(split)
        didx2sidx[-1][1] = len(flat_splits)
    logger.debug('doc_texts flattened')
    q_tokens = processes.map_async(tokenize_text, questions)
    logger.info('begin tokenizing...')
    logger.debug('doc ids are:(%s)' % ','.join([i for j in all_doc_ids for i in j]))
    s_tokens = processes.map_async(tokenize_text, flat_splits)
    q_tokens = q_tokens.get()
    s_tokens = s_tokens.get()
    # logger.info('q_tokens: %s' % q_tokens)
    # logger.info('s_tokens: %s' % s_tokens)
    t6 = time.time()
    logger.info('doc texts tokenized [time]: %.4f s' % (t6 - t4))

    for qid in range(len(questions)):
        q_text = q_tokens[qid].words()
        q_id = slugify(questions[qid])
        q_feat_file = os.path.join(save_dir, '%s.json' % q_id)
        if not os.path.exists(q_feat_file):
            para_length = len(q_text)
            counter = Counter(q_text)
            tf = [round(counter[w] * 1.0 / para_length, 6) for w in q_text]
            record = {
                'words': q_text,
                'pos': q_tokens[qid].pos(),
                'ner': q_tokens[qid].entities(),
                'tf': tf
            }
            with open(q_feat_file, 'w') as f:
                f.write(json.dumps(record, sort_keys=True))

        para_lens = []
        for rel_didx, did in enumerate(all_doc_ids[qid]):
            start, end = didx2sidx[did2didx[did]]
            for sidx in range(start, end):
                para_text = s_tokens[sidx].words()
                if len(q_text) > 0 and len(para_text) > 0:
                    para_lens.append(len(s_tokens[sidx].words()))
                    feat_file = os.path.join(save_dir, '%s.json' % did)
                    if not os.path.exists(feat_file):
                        para_length = len(para_text)
                        counter = Counter(para_text)
                        tf = [round(counter[w] * 1.0 / para_length, 6) for w in para_text]
                        record = {
                            'pos': s_tokens[sidx].pos(),
                            'ner': s_tokens[sidx].entities(),
                            'tf': tf,
                            'words': para_text
                        }
                        with open(feat_file, 'w') as f:
                            f.write(json.dumps(record, sort_keys=True))
        logger.debug('question_p: %s paragraphs: %s' % (questions[qid], para_lens))
    t7 = time.time()
    logger.info('paragraphs prepared [time]: %.4f s' % (t7 - t6))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('-i', '--index_path', type=str, default=None,
                        help='Path to wikipedia lucene index')
    parser.add_argument('--n-docs', type=int, default=150,
                        help="Number of docs to retrieve per query")
    parser.add_argument('--tokenizer', type=str, default='corenlp',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Question batching size')
    parser.add_argument('--ranker', type=str, default='lucene')
    parser.add_argument('-f', '--save_dir', type=str, default=None)
    parser.add_argument("-v", "--verbose", help="log more debug info",
                        action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    t0 = time.time()
    log_filename = ('_'.join(sys.argv) + time.strftime("%Y%m%d-%H%M%S")).replace('/', '_')
    logfile = logging.FileHandler('/tmp/%s.log' % log_filename, 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info('COMMAND: python %s' % ' '.join(sys.argv))

    ranker = LuceneRanker(index_path=args.index_path)

    logger.info('Loading queries from %s' % args.dataset)
    queries = []
    for line in open(args.dataset):
        data = json.loads(line)
        queries.append(data['question'])

    basename = os.path.splitext(os.path.basename(args.dataset))[0]

    annotators = set()
    annotators.add('pos')
    annotators.add('ner')
    tok_opts = {'annotators': annotators}
    tok_class = tokenizers.get_class(args.tokenizer)
    pool = ProcessPool(args.num_workers,
                       initializer=init,
                       initargs=(tok_class, tok_opts))

    batches = [queries[i: i + args.batch_size]
               for i in range(0, len(queries), args.batch_size)]
    for i, batch in enumerate(batches):
        batch_info = '-' * 5 + ' Batch %d/%d ' % (i + 1, len(batches)) + '-' * 5 + ' '
        start_query = batch[0]
        logger.info(batch_info + start_query)
        process_batch(ranker, batch, args.save_dir, args.n_docs, pool)

    logger.info('Total time: %.4f' % (time.time() - t0))
