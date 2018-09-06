#!/usr/bin/env python3
import argparse
import json
import os
from utils import exact_match_score, normalize, regex_match_score, get_rank

ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_file', type=str)
    parser.add_argument('-p', '--prediction_file', type=str)
    parser.add_argument('-sp', '--stop_perfect', type=str, default='',
                        help='stop perfectly if given a file path')
    parser.add_argument('-nr', '--no_regex', action='store_true', help='default to use regex match')
    parser.add_argument('-sl', '--stop_location', type=int, default=-1, help='manual stop location')

    args = parser.parse_args()
    answer_file = args.answer_file
    prediction_file = args.prediction_file
    match_func = exact_match_score if args.no_regex else regex_match_score

    if args.stop_perfect:
        with open(args.stop_perfect, 'w', encoding=ENCODING) as pf:
            for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                                  open(prediction_file, encoding=ENCODING)):
                data = json.loads(data_line)
                answer = [normalize(a) for a in data['answer']]
                prediction = json.loads(prediction_line)
                doc_predictions = sorted(prediction, key=lambda k: k['doc_score'], reverse=True)

                prediction_rank = get_rank(doc_predictions, answer, match_func)
                pf.write(json.dumps(doc_predictions[:prediction_rank]) + '\n')

    if args.stop_location > 0:
        prediction_dir = os.path.dirname(prediction_file)
        out_dir = os.path.join(prediction_dir, 'fixed_stop')
        os.makedirs(out_dir, exist_ok=True)
        prediction_file_base= os.path.splitext(os.path.basename(prediction_file))[0]
        out_file = os.path.join(out_dir, prediction_file_base + '.stop' + str(args.stop_location) + '.txt')

        with open(out_file, 'w', encoding=ENCODING) as sf:
            for prediction_line in open(prediction_file, encoding=ENCODING):
                prediction = json.loads(prediction_line)
                doc_predictions = sorted(prediction, key=lambda k: k['doc_score'], reverse=True)
                sf.write(json.dumps(doc_predictions[:args.stop_location]) + '\n')
