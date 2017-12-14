#!/usr/bin/env python3
import json
import argparse
import os
from collections import OrderedDict
from drqa.retriever.utils import normalize
from drqa.pipeline import DEFAULTS
from drqa.reader.utils import exact_match_score, metric_max_over_ground_truths
ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_file',
                        default='../../data/earlystopping/SQuAD-v1.1-dev-10-multitask-pipeline.preds')
    parser.add_argument('-a', '--answer_file', default='../../data/datasets/SQuAD-v1.1-dev-10.txt')
    parser.add_argument('-r', '--record_file',
                        default='../../data/earlystopping/records-10.pkl')
    args = parser.parse_args()

    doc_missing_count = 0
    no = 0
    record_path = os.path.splitext(args.record_file)[0] + '.txt'
    answer_file = args.answer_file
    prediction_file = args.prediction_file
    with open(record_path, 'w') as f:
        for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                              open(prediction_file, encoding=ENCODING)):
            data = json.loads(data_line)
            question = data['question']
            answer = [normalize(a) for a in data['answer']]
            prediction = json.loads(prediction_line)
            for entry in prediction:
                doc_id = entry['doc_id']
                record = OrderedDict()
                record['q'] = question
                record['d_id'] = doc_id
                record['a'] = normalize(entry['span'])
                record['a_s'] = entry['span_score']
                record['d_s'] = entry['doc_score']
                exact_match = metric_max_over_ground_truths(exact_match_score, normalize(entry['span']), answer)
                record['stop'] = 1 if exact_match else 0

                doc_path = os.path.join(DEFAULTS['features'], '%s.json' % doc_id)
                if os.path.exists(doc_path):
                    doc_data = open(doc_path, encoding=ENCODING).read()
                    feature = json.loads(doc_data)

                    record['d_l'] = feature['l_p']
                    record['tf'] = feature['tf']
                    record['ner'] = feature['ner']
                    record['pos'] = feature['pos']
                else:
                    print('%s not exist!' % doc_path)
                    doc_missing_count += 1
                f.write(json.dumps(record, sort_keys=True) + '\n')
                no += 1
                print('processed %d records...' % no)

    print('%d docs not found' % doc_missing_count)

