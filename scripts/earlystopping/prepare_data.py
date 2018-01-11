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
from multiprocessing import Pool as ProcessPool

import numpy as np
import pickle as pk
import sys
import time
ENCODING = "utf-8"


def process_record(data_line_, prediction_line_, pos_gap_, neg_gap_, record_dir_):
    missing_count_ = 0
    total_count_ = 0
    stop_count_ = 0
    data = json.loads(data_line_)
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
        print('question feature file %s not exist!' % q_path)
        sys.stdout.flush()
        missing_count_ += 1
        return missing_count_, total_count_, stop_count_
    q_h_path = os.path.join(DEFAULTS['features'], '%s.npz' % q_id)

    if os.path.exists(q_h_path):
        q_h_data = np.load(q_h_path)
        q_h = q_h_data['q_hidden']
    else:
        print('question hidden file %s not exist!' % q_h_path)
        sys.stdout.flush()
        return missing_count_, total_count_, stop_count_
    answer = [normalize(a) for a in data['answer']]
    prediction = json.loads(prediction_line_)
    ranked_prediction = sorted(prediction, key=lambda k: k['doc_score'])
    found_correct = False
    all_n_p = []
    all_n_a = []
    all_p_hidden = []
    all_a_hidden = []
    all_p_scores = []
    all_a_scores = []
    for i, entry in enumerate(ranked_prediction):
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
            print('paragraph hidden file %s not exist!' % p_h_path)
            sys.stdout.flush()
            missing_count_ += 1
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
        record['hq'] = list(map(float, q_h))

        if not found_correct:
            found_correct = metric_max_over_ground_truths(exact_match_score, normalize(entry['span']), answer)

        if found_correct:
            if i % pos_gap_ == 0:
                record['stop'] = 1
                stop_count_ += 1
                write_record = True
            else:
                write_record = False
        else:
            if i % neg_gap_ == 0:
                record['stop'] = 0
                write_record = True
            else:
                write_record = False
        if write_record:
            record_path = os.path.join(record_dir_, '%s_%s.pkl' % (q_id, doc_id))
            with open(record_path, 'wb') as f:
                pk.dump(record, f)
            total_count_ += 1
    return missing_count_, total_count_, stop_count_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_file',
                        default='data/earlystopping/SQuAD-v1.1-dev-100-multitask-pipeline.preds')
    parser.add_argument('-a', '--answer_file', default='data/datasets/SQuAD-v1.1-dev-100.txt')
    parser.add_argument('-m', '--no_multiprocess', action='store_true', help='default to use multiprocessing')
    parser.add_argument('-ps', '--positive_scale', type=int, default=3, help='scale factor for positive samples')
    parser.add_argument('-ns', '--negative_scale', type=int, default=10, help='scale factor for negative samples')
    parser.add_argument('-r', '--record_dir', default=DEFAULTS['records'])

    args = parser.parse_args()

    missing_count = 0
    total_count = 0
    stop_count = 0

    answer_file = args.answer_file
    prediction_file = args.prediction_file
    record_dir = args.record_dir
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    s = time.time()
    if args.no_multiprocess:
        for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                              open(prediction_file, encoding=ENCODING)):
            missing, total, stop = process_record(data_line, prediction_line,
                                                  args.positive_scale, args.negative_scale, record_dir)
            missing_count += missing
            stop_count += stop
            total_count += total
            print('processed %d records...' % total_count)
            sys.stdout.flush()
    else:
        print('using multiprocessing...')
        result_handles = []
        async_pool = ProcessPool()
        for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                              open(prediction_file, encoding=ENCODING)):
            param = (data_line, prediction_line, args.positive_scale, args.negative_scale, record_dir)
            handle = async_pool.apply_async(process_record, param)
            result_handles.append(handle)
        for result in result_handles:
            missing, total, stop = result.get()
            missing_count += missing
            stop_count += stop
            total_count += total
            print('processed %d records, stop: %d' % (total_count, stop_count))
            sys.stdout.flush()

    e = time.time()
    print('%d records' % total_count)
    print('%d stop labels' % stop_count)
    print('%d docs not found' % missing_count)
    print('took %.4f s' % (e - s))
