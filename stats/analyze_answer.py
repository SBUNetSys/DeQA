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
    parser.add_argument('-p', '--prediction_file', default='data/earlystopping/SQuAD-v1.1-dev-multitask-pipeline.preds')
    parser.add_argument('-ans', '--answer_rank', action='store_true', help='default to use doc score rank')
    parser.add_argument('-r', '--regex', action='store_true', help='default to use exact match')
    parser.add_argument('-d', '--draw', action='store_true', help='default not output draw data')
    parser.add_argument('-s', '--stop_location', type=int, default=150, help='manual stop location')
    parser.add_argument('-t', '--top_n', type=int, default=150, help='print top n accuracy')

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
        prediction = sorted(prediction, key=lambda k: -k['doc_score'])
        top_prediction = prediction[:args.stop_location]
        if args.answer_rank:
            prediction = sorted(top_prediction, key=lambda k: -k['span_score'])
        doc_rank = get_rank(prediction, answer, args.regex)
        ranks.append(doc_rank)

    rank_counter = Counter(ranks)
    acc_rank = 0
    if args.draw:
        keys = range(1, 151, 1)
    else:
        keys = sorted(rank_counter.keys())

    top_n = args.top_n
    for rank in keys:
        num = rank_counter.get(rank, 0)
        rate = num / len(ranks) * 100
        acc_rank += num
        acc_rate = acc_rank / len(ranks) * 100

        if rank <= top_n:
            if args.draw:
                print('%.1f%%' % acc_rate)
            else:
                print('%s, %s, %.1f%%, %.1f%%' % (rank, rank_counter.get(rank), rate, acc_rate))
        else:
            left_rank = len(ranks) - acc_rank + num  # already deducted num right at the top_n, need to add it back
            left_rate = left_rank / len(ranks) * 100
            if args.draw:
                print()
            else:
                print('>%d, %s, %.1f%%, %.1f%%' % (top_n, left_rank, left_rate, 100.0))
            break
