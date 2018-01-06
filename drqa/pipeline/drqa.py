#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Full DrQA pipeline."""

import torch
import regex
import heapq
import math
import json
from collections import Counter
import os
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from ..reader.vector import batchify
from ..reader.data import ReaderDataset, SortedBatchSampler
from .. import reader
from .. import tokenizers
from ..tokenizers.tokenizer import Tokenizer
from . import DEFAULTS
import logging
from ..reader.utils import slugify
import time

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Multiprocessing functions to fetch and tokenize text
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# Main DrQA pipeline
# ------------------------------------------------------------------------------


class DrQA(object):
    # Target size for squashing short paragraphs together.
    # 0 = read every paragraph independently
    # infty = read all paragraphs together
    GROUP_LENGTH = 0

    def __init__(
            self,
            reader_model=None,
            embedding_file=None,
            tokenizer=None,
            fixed_candidates=None,
            batch_size=128,
            cuda=True,
            data_parallel=False,
            max_loaders=5,
            num_workers=None,
            ranker_config=None
    ):
        """Initialize the pipeline.

        Args:
            reader_model: model file from which to load the DocReader.
            embedding_file: if given, will expand DocReader dictionary to use
              all available pretrained embeddings.
            tokenizer: string option to specify tokenizer used on docs.
            fixed_candidates: if given, all predictions will be constrated to
              the set of candidates contained in the file. One entry per line.
            batch_size: batch size when processing paragraphs.
            cuda: whether to use the gpu.
            data_parallel: whether to use multile gpus.
            max_loaders: max number of async data loading workers when reading.
              (default is fine).
            num_workers: number of parallel CPU processes to use for tokenizing
              and post processing resuls.
            ranker_config: config for ranker.
        """
        self.batch_size = batch_size
        self.max_loaders = max_loaders
        self.fixed_candidates = fixed_candidates is not None
        self.cuda = cuda

        feat_dir = DEFAULTS['features']
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        logger.info('Initializing document ranker...')
        ranker_config = ranker_config or {}
        ranker_class = ranker_config.get('class', DEFAULTS['ranker'])
        ranker_opts = ranker_config.get('options', {})
        self.ranker = ranker_class(**ranker_opts)

        logger.info('Initializing document reader...')
        t0 = time.time()
        reader_model = reader_model or DEFAULTS['reader_model']
        self.reader = reader.DocReader.load(reader_model, normalize=False)
        t1 = time.time()
        logger.info('document reader model load [time]: %.4f s' % (t1 - t0))

        if embedding_file:
            logger.info('embedding_file')
            logger.info('Expanding dictionary...')
            words = reader.utils.index_embedding_words(embedding_file)
            added = self.reader.expand_dictionary(words)
            self.reader.load_embeddings(added, embedding_file)

        if cuda:
            logger.info('cuda')
            self.reader.cuda()
        t2 = time.time()
        logger.info('cuda initialized [time]: %.4f s' % (t2 - t1))

        if data_parallel:
            logger.info('data_parallel')
            self.reader.parallelize()

        logger.debug('tokenizer')
        if not tokenizer:
            tok_class = DEFAULTS['tokenizer']
        else:
            tok_class = tokenizers.get_class(tokenizer)

        logger.debug('annotators')
        annotators = tokenizers.get_annotators_for_model(self.reader)
        tok_opts = {'annotators': annotators}

        self.num_workers = num_workers
        self.processes = ProcessPool(
            num_workers,
            initializer=init,
            initargs=(tok_class, tok_opts, fixed_candidates)
        )

    def _split_doc(self, doc):
        """Given a doc, split it into chunks (by paragraph)."""
        curr = []
        curr_len = 0
        for split in regex.split(r'\n+', doc):
            split = split.strip()
            if len(split) == 0:
                continue
            # Maybe group paragraphs together until we hit a length limit
            if len(curr) > 0 and curr_len + len(split) > self.GROUP_LENGTH:
                yield ' '.join(curr)
                curr = []
                curr_len = 0
            curr.append(split)
            curr_len += len(split)
        if len(curr) > 0:
            yield ' '.join(curr)

    def _get_loader(self, data, num_loaders):
        """Return a pytorch data iterator for provided examples."""
        dataset = ReaderDataset(data, self.reader)
        sampler = SortedBatchSampler(
            dataset.lengths(),
            self.batch_size,
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_loaders,
            collate_fn=batchify,
            pin_memory=self.cuda,
        )
        return loader

    def process(self, query, candidates=None, top_n=1, n_docs=5,
                return_context=False):
        """Run a single query."""
        predictions = self.process_batch(
            [query], [candidates] if candidates else None,
            top_n, n_docs, return_context
        )
        return predictions[0]

    def process_batch(self, queries, candidates=None, top_n=1, n_docs=5,
                      return_context=False):
        """Run a batch of queries (more efficient)."""
        t3 = time.time()
        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d docs...' % n_docs)

        # Rank documents for queries.
        if len(queries) == 1:
            ranked = [self.ranker.closest_docs(queries[0], k=n_docs)]
        else:
            ranked = self.ranker.batch_closest_docs(queries, k=n_docs, num_workers=self.num_workers)

        t4 = time.time()
        logger.info('docs retrieved [time]: %.4f s' % (t4 - t3))
        all_docids, all_doc_scores, all_doc_texts = zip(*ranked)

        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        flat_docids, flat_doc_texts = zip(*{(d, t) for doc_ids, doc_texts in zip(all_docids, all_doc_texts)
                                            for d, t in zip(doc_ids, doc_texts)})

        # flat_docids = list({d for docids in all_docids for d in docids})
        did2didx = {did: didx for didx, did in enumerate(flat_docids)}
        # flat_doc_texts = list({t for doc_texts in all_doc_texts for t in doc_texts})
        # logger.info('doc_texts for top %d docs extracted' % n_docs)

        # Split and flatten documents. Maintain a mapping from doc (index in
        # flat list) to split (index in flat list).
        flat_splits = []
        didx2sidx = []
        for text in flat_doc_texts:
            splits = self._split_doc(text)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
            didx2sidx[-1][1] = len(flat_splits)
        t5 = time.time()
        logger.debug('doc_texts flattened')

        # Push through the tokenizers as fast as possible.
        q_tokens = self.processes.map_async(tokenize_text, queries)
        s_tokens = self.processes.map_async(tokenize_text, flat_splits)
        q_tokens = q_tokens.get()
        s_tokens = s_tokens.get()
        # logger.info('q_tokens: %s' % q_tokens)
        # logger.info('s_tokens: %s' % s_tokens)
        t6 = time.time()
        logger.info('doc texts tokenized [time]: %.4f s' % (t6 - t5))

        # Group into structured example inputs. Examples' ids represent
        # mappings to their question, document, and split ids.
        examples = []
        q_text = q_tokens[0].words()
        q_id = slugify(queries[0])
        q_feat_file = os.path.join(DEFAULTS['features'], '%s.json' % q_id)
        n_q = [0 for _ in Tokenizer.FEAT]
        if os.path.exists(q_feat_file):
            record = json.load(open(q_feat_file))
            q_ner = record['ner']
            q_pos = record['pos']
            for feat in q_ner + q_pos:
                n_q[Tokenizer.FEAT_DICT[feat]] += 1
        else:
            logger.warning('no ner and pos file: %s' % q_feat_file)
            return []

        para_lens = []
        p_pos = dict()
        p_ner = dict()
        for rel_didx, did in enumerate(all_docids[0]):
            start, end = didx2sidx[did2didx[did]]
            for sidx in range(start, end):
                para_text = s_tokens[sidx].words()
                if len(q_text) > 0 and len(para_text) > 0:
                    examples.append({
                        'id': (rel_didx, sidx),
                        'question': q_text,
                        'qlemma': q_tokens[0].lemmas(),
                        'document': para_text,
                        'lemma': s_tokens[sidx].lemmas(),
                        'doc_score': float(all_doc_scores[0][rel_didx])
                    })
                    para_lens.append(len(para_text))

                    feat_file = os.path.join(DEFAULTS['features'], '%s.json' % did)
                    if os.path.exists(feat_file):
                        record = json.load(open(feat_file))
                        p_ner[did] = record['ner']
                        p_pos[did] = record['pos']
                    else:
                        logger.warning('no ner and pos file: %s' % feat_file)

            logger.debug('question_p: %s paragraphs: %s' % (queries[0], para_lens))
        t7 = time.time()
        logger.info('paragraphs prepared [time]: %.4f s' % (t7 - t6))
        num_loaders = min(self.max_loaders, int(math.floor(len(examples) / 1e3)))
        all_predictions = []
        predictions = []

        all_n_p = []
        all_n_a = []
        all_p_hidden = []
        all_a_hidden = []
        all_p_scores = []
        all_a_scores = []
        for batch in self._get_loader(examples, num_loaders):
            ids = batch[-1][0]
            rel_didx, sidx = ids
            doc_id = all_docids[0][rel_didx]
            doc_score = float(all_doc_scores[0][rel_didx])

            handle = self.reader.predict(batch, async_pool=self.processes)
            start, end, a_score, q_hidden, p_hidden, a_hidden = handle.get()

            n_p = [0 for _ in Tokenizer.FEAT]
            n_a = [0 for _ in Tokenizer.FEAT]
            for feat in p_ner[doc_id] + p_pos[doc_id]:
                n_p[Tokenizer.FEAT_DICT[feat]] += 1
            for feat in p_ner[doc_id][start:end + 1] + p_pos[doc_id][start:end + 1]:
                n_a[Tokenizer.FEAT_DICT[feat]] += 1
            all_n_p.append(n_p)
            all_n_a.append(n_a)
            all_p_hidden.append(p_hidden)
            all_a_hidden.append(a_hidden)
            all_p_scores.append(doc_score)
            all_a_scores.append(a_score)

            # TODO: feed features into early stopping model and decide whether to stop or not
            prediction = {
                'doc_id': doc_id,
                'start': start,
                'end': end,
                'span': s_tokens[sidx].slice(start, end + 1).untokenize(),
                'doc_score': doc_score,
                'span_score': a_score,
            }
            if return_context:
                prediction['context'] = {
                    'text': s_tokens[sidx].untokenize(),
                    'start': s_tokens[sidx].offsets()[start][0],
                    'end': s_tokens[sidx].offsets()[end][1],
                }
            predictions.append(prediction)
        t8 = time.time()
        logger.info('paragraphs predicted [time]: %.4f s' % (t8 - t7))
        all_predictions.append(predictions[-1::-1])
        logger.info('%d queries processed [time]: %.4f s' %
                    (len(queries), time.time() - t3))

        return all_predictions
