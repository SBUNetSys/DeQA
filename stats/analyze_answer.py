#!/usr/bin/env python3
import json
import argparse
from collections import Counter
from drqa.retriever.utils import normalize
from drqa.reader.utils import exact_match_score, metric_max_over_ground_truths

ENCODING = "utf-8"


def get_rank(prediction_, answer_, show_score):
    doc_scores = sorted([float(e['doc_score']) for e in prediction_], reverse=True)
    for ans_rank_, entry in enumerate(prediction_):
        exact_match = metric_max_over_ground_truths(exact_match_score, normalize(entry['span']), answer_)
        if exact_match:
            if show_score:
                print(entry['span_score'])
            doc_score = float(entry['doc_score'])
            doc_rank_ = doc_scores.index(doc_score)
            return ans_rank_ + 1, doc_rank_ + 1
    return 151, 151


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_file', type=str, default='data/datasets/SQuAD-v1.1-dev.txt')
    parser.add_argument('-p', '--prediction_file',
                        default='data/earlystopping/SQuAD-v1.1-dev-multitask-pipeline.preds')
    parser.add_argument('-s', '--score', action='store_true')

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
        ans_rank, doc_rank = get_rank(prediction, answer, args.score)
        if ans_rank > doc_rank:
            proceed_count += 1
            # print('q:', question, 'a:', answer, ans_rank, doc_rank)
        ranks.append((ans_rank, doc_rank))

    rank_counter = Counter([r[0] for r in ranks])
    acc_rank = 0
    for rank in sorted(rank_counter.keys()):
        num = rank_counter.get(rank)
        rate = num / len(ranks) * 100
        acc_rank += num
        acc_rate = acc_rank / len(ranks) * 100
        print('%s, %s, %.1f%%, %.1f%%' % (rank, rank_counter.get(rank), rate, acc_rate))

    correct_count = acc_rank - rank_counter.get(151)
    print('ans_rank > doc_rank', proceed_count, correct_count, '%.1f%%' % (proceed_count / correct_count * 100))
