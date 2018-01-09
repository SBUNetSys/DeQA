#!/usr/bin/env python3
import json
import argparse
from collections import Counter
from drqa.retriever.utils import normalize
from drqa.reader.utils import exact_match_score, metric_max_over_ground_truths

ENCODING = "utf-8"


def get_rank(prediction_, answer_):
    for doc_rank_, entry in enumerate(prediction_):
        exact_match = metric_max_over_ground_truths(exact_match_score, normalize(entry['span']), answer_)
        if exact_match:
            return doc_rank_ + 1
    return 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_file',
                        default='data/earlystopping/SQuAD-v1.1-dev-100-multitask-pipeline.preds')

    args = parser.parse_args()
    prediction_file = args.prediction_file
    outfile = prediction_file + '.txt'
    with open(outfile, 'w') as f:
        for prediction_line in open(prediction_file, encoding=ENCODING):
            prediction = json.loads(prediction_line)
            ranked_prediction = sorted(prediction, key=lambda k: -k['doc_score'])
            new_prediction = []
            for rank_, entry in enumerate(ranked_prediction):
                entry['rank'] = rank_
                new_prediction.append(entry)
            f.write(json.dumps(new_prediction) + '\n')
