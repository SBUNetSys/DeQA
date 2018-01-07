#!/usr/bin/env python3
import json
import argparse
import os
from collections import OrderedDict
from drqa.retriever.utils import normalize
from drqa.pipeline import DEFAULTS
from drqa.reader.utils import exact_match_score, metric_max_over_ground_truths
from drqa.reader.utils import slugify, aggregate
from drqa.tokenizers.tokenizer import Tokenizer
import numpy as np
import pickle as pk
ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_file',
                        default='../../data/earlystopping/SQuAD-v1.1-dev-100-multitask-pipeline.preds')
    parser.add_argument('-a', '--answer_file', default='../../data/datasets/SQuAD-v1.1-dev-100.txt')

    args = parser.parse_args()

    doc_missing_count = 0
    no = 0
    answer_file = args.answer_file
    prediction_file = args.prediction_file
    if not os.path.exists(DEFAULTS['records']):
        os.makedirs(DEFAULTS['records'])

    stop_count = 0
    for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                          open(prediction_file, encoding=ENCODING)):
        data = json.loads(data_line)
        question = data['question']
        q_id = slugify(question)
        q_path = os.path.join(DEFAULTS['features'], '%s.json' % q_id)
        n_q = [0 for _ in Tokenizer.FEAT]
        if os.path.exists(q_path):
            q_data = open(q_path, encoding=ENCODING).read()
            record = json.loads(q_data)
            q_ner = record['ner']
            q_pos = record['pos']
            for feat in q_ner + q_pos:
                n_q[Tokenizer.FEAT_DICT[feat]] += 1
        else:
            print('%s not exist!' % q_path)
            exit(-1)
            doc_missing_count += 1
        q_h_path = os.path.join(DEFAULTS['features'], '%s.npz' % q_id)

        if os.path.exists(q_h_path):
            q_h_data = np.load(q_h_path)
            q_h = q_h_data['q_hidden']
        else:
            q_h = None
            print('%s not exist!' % q_h_path)
            exit(-1)
        answer = [normalize(a) for a in data['answer']]
        prediction = json.loads(prediction_line)
        ranked_prediction = sorted(prediction, key=lambda k: k['doc_score'])
        found_correct = False
        all_n_p = []
        all_n_a = []
        all_p_hidden = []
        all_a_hidden = []
        all_p_scores = []
        all_a_scores = []
        for entry in ranked_prediction:
            doc_id = entry['doc_id']
            start = int(entry['start'])
            end = int(entry['end'])
            doc_score = entry['doc_score']
            ans_score = entry['span_score']

            p_pos = dict()
            p_ner = dict()
            feat_file = os.path.join(DEFAULTS['features'], '%s.json' % doc_id)
            if os.path.exists(feat_file):
                record = json.load(open(feat_file))
                p_ner[doc_id] = record['ner']
                p_pos[doc_id] = record['pos']
            n_p = [0 for _ in Tokenizer.FEAT]
            n_a = [0 for _ in Tokenizer.FEAT]
            for feat in p_ner[doc_id] + p_pos[doc_id]:
                n_p[Tokenizer.FEAT_DICT[feat]] += 1
            for feat in p_ner[doc_id][start:end + 1] + p_pos[doc_id][start:end + 1]:
                n_a[Tokenizer.FEAT_DICT[feat]] += 1

            p_h_path = os.path.join(DEFAULTS['features'], '%s_%s.npz' % (q_id, doc_id))
            if os.path.exists(p_h_path):
                p_h_data = np.load(p_h_path)
                p_h = p_h_data['doc_hidden']
                a_h = p_h_data['ans_hidden']
            else:
                print('%s not exist!' % p_h_path)
                doc_missing_count += 1
                continue
            all_n_p.append(n_p)
            all_n_a.append(n_a)
            all_p_hidden.append(p_h)
            all_a_hidden.append(a_h)
            all_p_scores.append(doc_score)
            all_a_scores.append(ans_score)

            f_np = aggregate(all_n_p)
            f_na = aggregate(all_n_a)
            f_sp = aggregate(all_p_scores)
            f_sa = aggregate(all_a_scores)
            f_hp = aggregate(all_p_hidden)
            f_ha = aggregate(all_a_hidden)

            record = OrderedDict()
            record['q'] = question
            record['a'] = normalize(entry['span'])
            record['np'] = f_np
            record['na'] = f_na
            record['sp'] = f_sp
            record['sa'] = f_sa
            record['hp'] = f_hp
            record['ha'] = f_ha
            record['nq'] = list(map(float, n_q))
            record['hq'] = list(map(float, np.asarray(q_h, dtype=float)))

            exact_match = metric_max_over_ground_truths(exact_match_score, normalize(entry['span']), answer)
            if exact_match:
                record['stop'] = 1
                found_correct = True
                stop_count += 1
            else:
                record['stop'] = 0
            no += 1
            record_path = os.path.join(DEFAULTS['records'], '%s.pkl' % no)
            with open(record_path, 'wb') as f:
                pk.dump(record, f)
            # with open(record_path, 'w') as f:
            #     f.write(json.dumps(record, sort_keys=True))
            if no % 10 == 0:
                print('processed %d records...' % no)
            if found_correct:
                break

    print('processed %d records...' % no)
    print('stop count: %d' % stop_count)
    print('%d docs not found' % doc_missing_count)
