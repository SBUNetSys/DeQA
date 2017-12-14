#!/usr/bin/env python3
import json
import argparse
import os
from collections import Counter, OrderedDict
from drqa.retriever.utils import normalize
from drqa.pipeline import DEFAULTS
from drqa import DATA_DIR
import pickle

ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_file',
                        default='../../data/earlystopping/SQuAD-v1.1-dev-10-multitask-pipeline.preds')
    parser.add_argument('-a', '--answer_file', default='../../data/datasets/SQuAD-v1.1-dev-10.txt')
    parser.add_argument('-r', '--record_file',
                        default='../../data/earlystopping/records-10.pkl')
    args = parser.parse_args()
    questions = []
    answers = []
    for line in open(args.answer_file, encoding=ENCODING):
        data = json.loads(line)
        question = data['question']
        questions.append(question)
        answer = [normalize(a) for a in data['answer']]
        answers.append(answer)

    prediction_file = args.prediction_file
    predictions = []
    for line in open(prediction_file, encoding=ENCODING):
        prediction_data = json.loads(line)
        predictions.append(prediction_data)
        # ids = [d['doc_id'] for d in prediction_data]
        # counter = Counter(ids)
        # print(len(counter.keys()))
    records = []
    doc_missing_count = 0
    no = 0
    for question, answer, prediction in zip(questions, answers, predictions):
        for entry in prediction:
            print('processing %d doc...' % no)
            doc_id = entry['doc_id']
            record = OrderedDict()
            record['q'] = question
            record['d_id'] = doc_id
            record['a'] = entry['span']
            record['a_s'] = entry['span_score']
            record['d_s'] = entry['doc_score']
            record['stop'] = 1 if entry['span'].lower() in list(map(lambda x: x.lower(), answer)) else 0

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
            records.append(record)
            no += 1

    print('%d docs missing' % doc_missing_count)
    record_path = os.path.splitext(args.record_file)[0] + '.json'
    with open(record_path, 'w') as f:
        f.write(json.dumps(records, sort_keys=True))

    record_path = os.path.splitext(args.record_file)[0] + '.pkl'
    with open(record_path, 'wb') as f:
        pickle.dump(records, f)
