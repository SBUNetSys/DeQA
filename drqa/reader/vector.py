#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

import logging
import time
from collections import Counter

import torch

logger = logging.getLogger(__name__)


def pad_char(x, max_len=None):
    """
    pad input 2d array (list of list), if max_len is given, may truncate inner 1d array:
    e.g.
    x = [[3, 46, 73, 43, 46],
         [19, 65, 64],
         [2, 82, 55, 4],
         [18, 82, 76, 18, 82, 28, 82, 27, 2, 82, 36],
         [2, 46, 82],
         [25, 65, 39],
         [55, 2]]
    pad_array(x, 5) will be:
        [[3, 46, 73, 43, 46],
         [19, 65, 64, 0, 0],
         [2, 82, 55, 4, 0],
         [18, 82, 76, 18, 82],
         [2, 46, 82, 0, 0],
         [25, 65, 39, 0, 0],
         [55, 2, 0, 0, 0]]
    while pad_array(x) will be:
        [[3, 46, 73, 43, 46, 0, 0, 0, 0, 0, 0],
         [19, 65, 64, 0, 0, 0, 0, 0, 0, 0, 0],
         [2, 82, 55, 4, 0, 0, 0, 0, 0, 0, 0],
         [18, 82, 76, 18, 82, 28, 82, 27, 2, 82, 36],
         [2, 46, 82, 0, 0, 0, 0, 0, 0, 0, 0],
         [25, 65, 39, 0, 0, 0, 0, 0, 0, 0, 0],
         [55, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    :param x: input 2d array
    :param max_len: truncate len
    :return: padded list in rectangular format
    """
    max_length = max(len(row) for row in x)
    return [row[:max_len] + ['<NULL>'] * ((max_len if max_len else max_length) - len(row)) for row in x]


def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    t1 = time.time()

    args = model.args
    word_dict = model.word_dict
    char_dict = model.char_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    # qc = [[c for c in qw] for qw in ex['question']]
    # dc = [[c for c in dw] for dw in ex['document']]
    qc = pad_char(ex['question_char'], args.char_limit)
    dc = pad_char(ex['document_char'], args.char_limit)
    # if args.model_type == 'qanet':
    question_char = torch.LongTensor([[char_dict[c] for c in w] for w in qc])
    document_char = torch.LongTensor([[char_dict[c] for c in w] for w in dc])
    # else:
    #     # FIXME for rnet and mnemonic reader
    #     question_char = torch.LongTensor([char_dict[w[0]] for w in ex['question_char']])
    #     document_char = torch.LongTensor([char_dict[w[0]] for w in ex['document_char']])

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_exact_match:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_cased']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_uncased']] = 1.0
            if q_lemma and 'in_lemma' in feature_dict and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    t2 = time.time()
    # logger.debug('vectorize [time]: %.4f s' % (t2 - t1))
    # Maybe return without target
    if 'answers' not in ex:
        return document, document_char, features, question, question_char, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert (len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]

    return document, document_char, features, question, question_char, start, end, ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 5
    NUM_TARGETS = 2
    NUM_EXTRA = 1
    # t1 = time.time()

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    doc_chars = [ex[1] for ex in batch]
    features = [ex[2] for ex in batch]
    questions = [ex[3] for ex in batch]
    question_chars = [ex[4] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    char_len = len(doc_chars[0][0])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_c = torch.LongTensor(len(docs), max_length, char_len).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])
    for i, c in enumerate(doc_chars):
        x1_c[i, :c.size(0)].copy_(c)

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_c = torch.LongTensor(len(questions), max_length, char_len).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)
    for i, c in enumerate(question_chars):
        # logger.info('c0: %s' % c.size(0))
        # logger.info('c1: %s' % c.size(1))
        x2_c[i, :c.size(0)].copy_(c)
        # logger.info('x2_ci: %s' % x2_c[i])
    # logger.info('x2_c: %s' % x2_c)

    # t2 = time.time()
    # logger.debug('batchify [time]: %.4f s' % (t2 - t1))
    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][NUM_INPUTS]):
            y_s = torch.cat([ex[NUM_INPUTS] for ex in batch])
            y_e = torch.cat([ex[NUM_INPUTS + 1] for ex in batch])
        else:
            y_s = [ex[NUM_INPUTS] for ex in batch]
            y_e = [ex[NUM_INPUTS + 1] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')
    # logger.info('y_s: %s' % y_s)
    # logger.info('y_e: %s' % y_e)

    return x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_mask, y_s, y_e, ids
