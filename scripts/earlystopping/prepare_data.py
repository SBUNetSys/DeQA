#!/usr/bin/env python3
import json
import argparse
import os
from collections import OrderedDict
from drqa.retriever.utils import normalize
from drqa.pipeline import DEFAULTS
from drqa.reader.utils import exact_match_score, metric_max_over_ground_truths
from drqa.reader.utils import slugify
ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_file',
                        default='../../data/earlystopping/SQuAD-v1.1-dev-100-multitask-pipeline.preds')
    parser.add_argument('-a', '--answer_file', default='../../data/datasets/SQuAD-v1.1-dev-100.txt')
    parser.add_argument('-r', '--record_file',
                        default='../../data/earlystopping/records-100.txt')
    args = parser.parse_args()

    doc_missing_count = 0
    no = 0
    record_path = args.record_file
    answer_file = args.answer_file
    prediction_file = args.prediction_file
    max_q_len = 0
    max_p_len = 0
    with open(record_path, 'w') as f:
        for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                              open(prediction_file, encoding=ENCODING)):
            data = json.loads(data_line)
            question = data['question']
            q_id = slugify(question)
            q_path = os.path.join(DEFAULTS['features'], '%s.json' % q_id)
            q_feature = None
            if os.path.exists(q_path):
                q_data = open(q_path, encoding=ENCODING).read()
                q_feature = json.loads(q_data)
            else:
                print('%s not exist!' % q_path)
                exit(-1)
                doc_missing_count += 1
            answer = [normalize(a) for a in data['answer']]
            prediction = json.loads(prediction_line)
            ranked_prediction = sorted(prediction, key=lambda k: k['doc_score'])
            found_correct = False
            for entry in ranked_prediction:
                doc_id = entry['doc_id']
                record = OrderedDict()
                record['q'] = question
                record['a'] = normalize(entry['span'])
                record['a_s'] = entry['span_score']
                record['d_s'] = entry['doc_score']
                if found_correct:
                    record['stop'] = 1
                else:
                    exact_match = metric_max_over_ground_truths(exact_match_score, normalize(entry['span']), answer)
                    if exact_match:
                        found_correct = True
                        record['stop'] = 1
                    else:
                        record['stop'] = 0

                if len(q_feature['idx']) > max_q_len:
                    max_q_len = len(q_feature['idx'])

                record['q_ner'] = q_feature['ner']
                record['q_pos'] = q_feature['pos']

                s = entry['start']
                e = entry['end']
                record['a_loc'] = [int(s), int(e)]
                record['doc_id'] = doc_id

                doc_path = os.path.join(DEFAULTS['features'], '%s.json' % doc_id)
                if os.path.exists(doc_path):
                    doc_data = open(doc_path, encoding=ENCODING).read()
                    feature = json.loads(doc_data)
                    if len(feature['idx']) > max_p_len:
                        max_p_len = len(feature['idx'])
                    # a_idx = feature['idx'][int(s):int(e) + 1]
                    # record['a_idx'] = a_idx
                else:
                    print('%s not exist!' % doc_path)
                    doc_missing_count += 1
                f.write(json.dumps(record, sort_keys=True) + '\n')
                no += 1
                if no % 100 == 0:
                    print('processed %d records...' % no)

    print('max_q_len:', max_q_len, 'max_p_len:', max_p_len)
    print('%d docs not found' % doc_missing_count)
