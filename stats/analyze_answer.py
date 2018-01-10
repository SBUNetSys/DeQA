#!/usr/bin/env python3
import json
import argparse
from collections import Counter
from drqa.retriever.utils import normalize
from drqa.reader.utils import exact_match_score, regex_match_score, metric_max_over_ground_truths

ENCODING = "utf-8"


def get_rank(prediction_, answer_, use_regex_=False):
    for rank_, entry in enumerate(prediction_):
        if use_regex_:
            match_fn = regex_match_score
        else:
            match_fn = exact_match_score
        exact_match = metric_max_over_ground_truths(match_fn, normalize(entry['span']), answer_)
        if exact_match:
            return rank_ + 1
    return 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_file', type=str, default='data/datasets/SQuAD-v1.1-dev.txt')
    parser.add_argument('-p', '--prediction_file',
                        default='data/earlystopping/SQuAD-v1.1-dev-multitask-pipeline.preds')
    parser.add_argument('-ans', '--answer_rank', action='store_true', help='default to use doc score rank')
    parser.add_argument('-r', '--regex', action='store_true', help='default to use exact match')
    args = parser.parse_args()
    answer_file = args.answer_file
    prediction_file = args.prediction_file
    ranks = []
    proceed_count = 0
    for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                          open(prediction_file, encoding=ENCODING)):
        data = json.loads(data_line)
        question = data['question']
        answer = [normalize(a) for a in data['answer']]
        prediction = json.loads(prediction_line)
        if args.answer_rank:
            prediction = sorted(prediction, key=lambda k: -k['span_score'])
        else:
            prediction = sorted(prediction, key=lambda k: -k['doc_score'])
        doc_rank = get_rank(prediction, answer, args.regex)
        ranks.append(doc_rank)

    rank_counter = Counter(ranks)
    acc_rank = 0
    for rank in sorted(rank_counter.keys()):
        num = rank_counter.get(rank)
        rate = num / len(ranks) * 100
        acc_rank += num
        acc_rate = acc_rank / len(ranks) * 100
        print('%s, %s, %.1f%%, %.1f%%' % (rank, rank_counter.get(rank), rate, acc_rate))

