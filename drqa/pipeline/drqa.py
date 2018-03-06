#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Full DrQA pipeline."""

import heapq
import json
import logging
import math
import os
import time
from collections import Counter
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

import regex
import torch

from . import DEFAULTS
from .StoppingModel import EarlyStoppingModel
from .. import reader
from .. import tokenizers
from ..reader.data import ReaderDataset, SortedBatchSampler
from ..reader.utils import slugify, aggregate
from ..reader.vector import batchify
from ..tokenizers.tokenizer import Tokenizer

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
            ranker=None,
            features_dir=None,
            et_threshold=None,
            stop_rank=None
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
        """
        self.batch_size = batch_size
        self.max_loaders = max_loaders
        self.fixed_candidates = fixed_candidates is not None
        self.cuda = cuda

        self.features_dir = features_dir

        logger.info('Initializing document ranker...')
        self.ranker = ranker

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

        annotators = tokenizers.get_annotators_for_model(self.reader)
        tok_opts = {'annotators': annotators}

        logger.debug('tokenizer')
        if not tokenizer:
            tok_class = DEFAULTS['tokenizer']
        else:
            tok_class = tokenizers.get_class(tokenizer)

        logger.debug('annotators')

        self.num_workers = num_workers
        self.processes = ProcessPool(num_workers,
                                     initializer=init,
                                     initargs=(tok_class, tok_opts, fixed_candidates))
        self.stop_rank = stop_rank
        if et_threshold:
            self.et_threshold = et_threshold if 0 < et_threshold < 1 else 0.5
            logger.info('Initializing early stopping model...')
            et_model = DEFAULTS['linear_model']
            self.et_model = EarlyStoppingModel.load(et_model)
            logger.info('early stopping model (et threshold: %s) loaded.' % self.et_threshold)
        else:
            self.et_threshold = None

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

    def process_single(self, query, top_n=1, n_docs=5,
                       return_context=False):
        """Run a single query."""
        predictions = self.process_batch(
            [query],
            top_n, n_docs, return_context
        )
        return predictions[0]

    def process(self, query, top_n=1, n_docs=5):
        if self.stop_rank or self.et_threshold:
            predictions = self.process_batch_et(query, n_docs)
        else:
            predictions = self.process_batch(query, top_n=top_n, n_docs=n_docs)
        return predictions

    def process_batch(self, queries, top_n=1, n_docs=5,
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
        for qidx in range(len(queries)):
            q_text = q_tokens[qidx].words()
            para_lens = []
            for rel_didx, did in enumerate(all_docids[qidx]):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                    para_text = s_tokens[sidx].words()
                    if len(q_text) > 0 and len(para_text) > 0:
                        examples.append({
                            'id': (qidx, rel_didx, sidx),
                            'question': q_text,
                            'qlemma': q_tokens[qidx].lemmas(),
                            'document': para_text,
                            'lemma': s_tokens[sidx].lemmas(),
                            'pos': s_tokens[sidx].pos(),
                            'ner': s_tokens[sidx].entities(),
                            'doc_score': float(all_doc_scores[qidx][rel_didx])
                        })
                        para_lens.append(len(s_tokens[sidx].words()))
            logger.debug('question_p: %s paragraphs: %s' % (queries[qidx], para_lens))
        t7 = time.time()
        logger.info('paragraphs prepared [time]: %.4f s' % (t7 - t6))

        result_handles = []
        num_loaders = min(self.max_loaders, int(math.floor(len(examples) / 1e3)))
        for batch in self._get_loader(examples, num_loaders):
            handle = self.reader.predict(batch, async_pool=self.processes)
            result_handles.append((handle, batch[-1], batch[0].size(0)))

        t8 = time.time()
        logger.info('paragraphs predicted [time]: %.4f s' % (t8 - t7))

        # Iterate through the predictions, and maintain priority queues for
        # top scored answers for each question in the batch.
        queues = [[] for _ in range(len(queries))]
        for result, ex_ids, batch_size in result_handles:
            s, e, score, entropy, prob = result.get()
            for i in range(batch_size):
                # We take the top prediction per split.
                if len(score[i]) > 0:
                    item = (score[i][0], ex_ids[i], s[i][0], e[i][0], entropy[i][0], prob[i][0])
                    queue = queues[ex_ids[i][0]]
                    if len(queue) < top_n:
                        heapq.heappush(queue, item)
                    else:
                        heapq.heappushpop(queue, item)

        logger.info('answers processed...')
        # Arrange final top prediction data.
        all_predictions = []
        for queue in queues:
            predictions = []
            while len(queue) > 0:
                score, (qidx, rel_didx, sidx), s, e, entropy, prob = heapq.heappop(queue)
                prediction = {
                    'doc_id': all_docids[qidx][rel_didx],
                    'start': str(s),
                    'end': str(e),
                    'span': s_tokens[sidx].slice(s, e + 1).untokenize(),
                    'doc_score': float(all_doc_scores[qidx][rel_didx]),
                    'span_score': float(score),
                    'entropy': float(entropy),
                    'prob': float(prob)
                }
                if return_context:
                    prediction['context'] = {
                        'text': s_tokens[sidx].untokenize(),
                        'start': s_tokens[sidx].offsets()[s][0],
                        'end': s_tokens[sidx].offsets()[e][1],
                    }
                predictions.append(prediction)
            all_predictions.append(predictions[-1::-1])

        logger.info('%d queries processed [time]: %.4f s' %
                    (len(queries), time.time() - t3))

        return all_predictions

    def process_batch_et(self, queries, n_docs):
        """Run a batch of queries (more efficient)."""
        t3 = time.time()
        logger.info('ET Processing %d queries...' % len(queries))
        logger.info('ET Retrieving top %d docs...' % n_docs)

        # Rank documents for queries.
        if len(queries) == 1:
            ranked = [self.ranker.closest_docs(queries[0], k=n_docs)]
        else:
            ranked = self.ranker.batch_closest_docs(queries, k=n_docs, num_workers=self.num_workers)

        t4 = time.time()
        logger.info('ET docs retrieved [time]: %.4f s' % (t4 - t3))
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
        logger.debug('ET doc_texts flattened')

        # Push through the tokenizers as fast as possible.
        q_tokens = self.processes.map_async(tokenize_text, queries)
        s_tokens = self.processes.map_async(tokenize_text, flat_splits)
        q_tokens = q_tokens.get()
        s_tokens = s_tokens.get()
        # logger.info('q_tokens: %s' % q_tokens)
        # logger.info('s_tokens: %s' % s_tokens)
        t6 = time.time()
        logger.info('ET doc texts tokenized [time]: %.4f s' % (t6 - t5))

        # Group into structured example inputs. Examples' ids represent
        # mappings to their question, document, and split ids.
        examples = []
        q_text = q_tokens[0].words()
        q_id = slugify(queries[0])
        q_feat_file = os.path.join(self.features_dir, '%s.json' % q_id)
        n_q = [0 for _ in Tokenizer.FEAT]
        if os.path.exists(q_feat_file):
            record = json.load(open(q_feat_file))
            q_ner = record['ner']
            q_pos = record['pos']
            for feat in q_ner + q_pos:
                n_q[Tokenizer.FEAT_DICT[feat]] += 1
        else:
            logger.warning('no question ner and pos file: %s' % q_feat_file)

        f_nq = list(map(float, n_q))
        para_lens = []
        p_pos = dict()
        p_ner = dict()
        for rel_didx, did in enumerate(all_docids[0]):
            start, end = didx2sidx[did2didx[did]]
            for sidx in range(start, end):
                para_text = s_tokens[sidx].words()
                if len(q_text) > 0 and len(para_text) > 10:
                    examples.append({
                        'id': (rel_didx, sidx),
                        'question': q_text,
                        'qlemma': q_tokens[0].lemmas(),
                        'document': para_text,
                        'lemma': s_tokens[sidx].lemmas(),
                        'doc_score': float(all_doc_scores[0][rel_didx])
                    })
                    para_lens.append(len(para_text))

                    feat_file = os.path.join(self.features_dir, '%s.json' % did)
                    if os.path.exists(feat_file):
                        record = json.load(open(feat_file))
                        p_ner[did] = record['ner']
                        p_pos[did] = record['pos']
                    else:
                        logger.warning('no paragraph ner and pos file: %s' % feat_file)

            logger.debug('question_p: %s paragraphs: %s' % (queries[0], para_lens))
        t7 = time.time()
        logger.info('paragraphs prepared [time]: %.4f s' % (t7 - t6))
        num_loaders = min(self.max_loaders, int(math.floor(len(examples) / 1e3)))
        all_predictions = []
        predictions = []

        all_n_p = []
        all_n_a = []
        all_p_scores = []
        processed_count = 0
        for batch in self._get_loader(examples, num_loaders):
            handle = self.reader.predict(batch, async_pool=self.processes)
            starts, ends, ans_scores = handle.get()
            starts = [s[0] for s in starts]
            ends = [e[0] for e in ends]
            ans_scores = [float(a[0]) for a in ans_scores]

            doc_ids = [all_docids[0][ids_[0]] for ids_ in batch[-1]]
            doc_scores = [float(all_doc_scores[0][ids_[0]]) for ids_ in batch[-1]]
            sids = [ids_[1] for ids_ in batch[-1]]
            all_np = []
            all_na = []
            for doc_id, start, end in zip(doc_ids, starts, ends):
                n_p = [0 for _ in Tokenizer.FEAT]
                for feat in p_ner[doc_id] + p_pos[doc_id]:
                    n_p[Tokenizer.FEAT_DICT[feat]] += 1
                n_a = [0 for _ in Tokenizer.FEAT]
                for feat in p_ner[doc_id][start:end + 1] + p_pos[doc_id][start:end + 1]:
                    n_a[Tokenizer.FEAT_DICT[feat]] += 1
                all_np.append(n_p)
                all_na.append(n_a)

            all_n_p.extend(all_np)
            all_n_a.extend(all_na)
            all_p_scores.extend(doc_scores)

            f_d = (doc_ids, sids)
            f_n = (all_p_scores, all_n_p, all_n_a)
            r_s = (starts, ends, ans_scores)
            processed_count += len(ans_scores)
            stop, batch_predictions = self.batch_predict_stop(f_d, f_n, r_s, s_tokens, f_nq,
                                                              processed_count, self.stop_rank)
            predictions.extend(batch_predictions)
            if stop:
                break
            else:
                continue
        t8 = time.time()
        logger.info('paragraphs predicted [time]: %.4f s' % (t8 - t7))
        all_predictions.append(predictions[-1::-1])
        logger.info('%d queries processed [time]: %.4f s' %
                    (len(queries), time.time() - t3))

        return all_predictions

    def batch_predict_stop(self, f_d, f_n, r_s, s_tokens, f_nq, p_count=None, stop_rank=None):
        doc_ids, sids = f_d
        batch_size = len(doc_ids)
        all_s_p, all_np, all_na = f_n
        doc_scores, f_np, f_na = all_s_p[-batch_size:], all_np[-batch_size:], all_na[-batch_size:]
        starts, ends, ans_scores = r_s
        predictions_ = []
        should_stop = False
        for i, item in enumerate(zip(doc_ids, sids, doc_scores, starts, ends, ans_scores)):
            doc_id, sid, doc_score, start, end, a_score = item
            prediction = {
                'doc_id': doc_id,
                'start': int(start),
                'end': int(end),
                'span': s_tokens[sid].slice(start, end + 1).untokenize(),
                'doc_score': doc_score,
                'span_score': a_score,
            }
            predictions_.append(prediction)
            if stop_rank:
                if i + p_count >= stop_rank:
                    should_stop = True
                    break
            else:
                loc = - batch_size + i + 1 if - batch_size + i + 1 else None
                sp = all_s_p[: loc]
                np = all_np[: loc]
                na = all_na[: loc]
                f_sp = aggregate(sp)
                f_np = aggregate(np)
                f_na = aggregate(na)
                et_input = torch.FloatTensor(f_sp + f_nq + f_np + f_na)
                et_prob = self.et_model.predict(et_input, prob=True)
                if et_prob > self.et_threshold:
                    should_stop = True
                    break
        return should_stop, predictions_
