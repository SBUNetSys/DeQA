#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Follows official evaluation script for v1.1 of the SQuAD dataset."""

import argparse
import json
from drqa.retriever.utils import normalize
from drqa.reader.utils import (
    exact_match_score,
    regex_match_score,
    metric_max_over_ground_truths
)


def evaluate(dataset_file, prediction_file, regex=False):
    print('-' * 50)
    print('Dataset: %s' % dataset_file)
    print('Predictions: %s' % prediction_file)

    answers = []
    for line in open(args.dataset):
        data = json.loads(line)
        answer = [normalize(a) for a in data['answer']]
        answers.append(answer)

    predictions = []
    with open(prediction_file) as f:
        for line in f:
            data = json.loads(line)
            if len(data):
                prediction = normalize(data[0]['span'])
            else:
                prediction = ''
            predictions.append(prediction)

    exact_match = 0
    for i in range(len(predictions)):
        match_fn = regex_match_score if regex else exact_match_score
        exact_match += metric_max_over_ground_truths(
            match_fn, predictions[i], answers[i]
        )
    total = len(predictions)
    exact_match = 100.0 * exact_match / total
    print({'exact_match': exact_match})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='/Users/qqcao/GitRepos/DrQA-pria/data/datasets/CuratedTrec-test.txt')
    # parser.add_argument('--predictions', type=str, default='/Users/qqcao/GitRepos/DrQA-pria/data/datasets/mul/CuratedTrec-test.preds.txt')
    # parser.add_argument('--regex', action='store_false')
    parser.add_argument('--dataset', type=str, default='/Users/qqcao/GitRepos/DrQA-pria/data/datasets/SQuAD-v1.1-dev.txt')
    parser.add_argument('--predictions', type=str, default='/Users/qqcao/GitRepos/DrQA-pria/data/datasets/mul/SQuAD-v1.1-dev.preds.txt')
    parser.add_argument('--regex', action='store_true')
    args = parser.parse_args()
    evaluate(args.dataset, args.predictions, args.regex)
