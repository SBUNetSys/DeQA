#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DrQA reader utilities."""

import json
import time
import logging
import string
import regex as re
import unicodedata
from collections import Counter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


def load_data(args, filename, skip_no_answer=False):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]

    # Make case insensitive?
    if args.uncased_question or args.uncased_doc:
        for ex in examples:
            if args.uncased_question:
                ex['question'] = [w.lower() for w in ex['question']]
            if args.uncased_doc:
                ex['document'] = [w.lower() for w in ex['document']]

    # Skip unparsed (start/end) examples
    if skip_no_answer:
        examples = [ex for ex in examples if len(ex['answers']) > 0]

    return examples


def load_text(filename):
    """Load the paragraphs only of a SQuAD dataset. Store as qid -> text."""
    # Load JSON file
    with open(filename) as f:
        examples = json.load(f)['data']

    texts = {}
    for article in examples:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                texts[qa['id']] = paragraph['context']
    return texts


def load_answers(filename):
    """Load the answers only of a SQuAD dataset. Store as qid -> [answers]."""
    # Load JSON file
    with open(filename) as f:
        examples = json.load(f)['data']

    ans = {}
    for article in examples:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                ans[qa['id']] = list(map(lambda x: x['text'], qa['answers']))
    return ans


# ------------------------------------------------------------------------------
# Evaluation. Follows official evalutation script for v1.1 of the SQuAD dataset.
# ------------------------------------------------------------------------------
class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    NER = ['O', 'MISC', 'PERSON', 'LOCATION', 'DATE', 'NUMBER', 'ORGANIZATION',
           'SET', 'DURATION', 'ORDINAL', 'PERCENT', 'MONEY', 'TIME', ]

    POS = ['RB', ',', 'DT', 'NN', 'VBZ', 'JJ', '.', 'IN', 'NNP', 'POS', 'CC', 'VBG',
           'PRP', 'NNS', 'VBN', '``', 'NNPS', "''", 'TO', 'WRB', 'VBD', 'CD', '-LRB-',
           'WDT', '-RRB-', 'RBS', 'VBP', 'VB', 'JJS', ':', 'PRP$', 'WP', 'JJR', '$', 'RP',
           'MD', 'EX', '#', 'RBR', 'FW', 'WP$', 'UH', 'PDT', 'SYM', 'LS']
    FEAT = NER + POS
    NER_DICT = {f: i for i, f in enumerate(NER)}
    POS_DICT = {f: i for i, f in enumerate(POS)}
    FEAT_DICT = {f: i for i, f in enumerate(FEAT)}


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata
    value = unicodedata.normalize('NFKD', value)
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    return re.sub('[-\s]+', '-', value)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def regex_match_score(prediction, pattern):
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE
        )
    except BaseException:
        logger.warn('Regular expression failed to compile: %s' % pattern)
        return False
    return compiled.match(prediction) is not None


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


def aggregate(data_):
    """
    compute aggregated statistical feature for data
    :param data_: input data, vector or matrix
    :return: max, mean, std, var
    """
    import numpy as np
    features = np.max(data_, 0).flatten().tolist() + np.mean(data_, 0).flatten().tolist() \
               + np.std(data_, 0).flatten().tolist() + np.var(data_, 0).flatten().tolist()
    return features


def get_rank(prediction_, answer_):
    for doc_rank_, entry in enumerate(prediction_):
        exact_match = metric_max_over_ground_truths(exact_match_score, normalize(entry['span']), answer_)
        if exact_match:
            return doc_rank_ + 1
    return 1000
