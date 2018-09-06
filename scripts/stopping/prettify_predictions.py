#!/usr/bin/env python3
import argparse
import json
import os
from utils import exact_match_score, normalize, regex_match_score, get_rank

ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_file', type=str)
    parser.add_argument('-p', '--prediction_file', type=str, )
    parser.add_argument('-nr', '--no_regex', action='store_true', help='default to use regex match')

    args = parser.parse_args()
    answer_file = args.answer_file
    prediction_file = args.prediction_file
    match_func = exact_match_score if args.no_regex else regex_match_score

    question_count = 1
    prediction_file_base = os.path.splitext(os.path.basename(prediction_file))[0]
    prediction_dir = os.path.dirname(prediction_file)

    out_file = os.path.join(prediction_dir, prediction_file_base + '.readable.txt')

    with open(out_file, 'w', encoding=ENCODING) as f:
        for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                              open(prediction_file, encoding=ENCODING)):
            data = json.loads(data_line)
            question = data['question']
            answer = [normalize(a) for a in data['answer']]
            prediction = json.loads(prediction_line)
            doc_predictions = sorted(prediction, key=lambda k: k['doc_score'], reverse=True)
            doc_rank = get_rank(doc_predictions, answer, match_func)

            ans_rank = get_rank(sorted(prediction, key=lambda k: k['span_score'], reverse=True), answer, match_func)
            qa_str = 'q_{}: {}\n'.format(question_count, question)
            qa_str += 'ans_rank: {}, doc_rank: {}, answer: {}\n'.format(ans_rank, doc_rank, '; '.join(answer))

            for d_no, ans_prediction in enumerate(doc_predictions, 1):
                qa_str += '\tdoc_{:3s}: {:12s}, d_score: {:.4f}, a_score: {:.4f}, ans: {:20s}, s: {}, e: {}\n'.format(
                    str(d_no), ans_prediction['doc_id'], ans_prediction['doc_score'], ans_prediction['span_score'],
                    ans_prediction['span'], ans_prediction['start'], ans_prediction['end'])

            f.write(qa_str + '\n')
            question_count += 1
