#!/usr/bin/env python3
import json
import argparse
import os
from collections import OrderedDict
from utils import normalize
from utils import exact_match_score, regex_match_score, get_rank
from utils import slugify, aggregate, aggregate_ans
from utils import Tokenizer
from multiprocessing import Pool as ProcessPool

# import numpy as np
import pickle as pk
import sys
import time

ENCODING = "utf-8"


def process_record(data_line_, prediction_line_, neg_gap_, feature_dir_, record_dir_, match_fn):
    missing_count_ = 0
    total_count_ = 0
    stop_count_ = 0
    data = json.loads(data_line_)
    question = data['question']
    q_id = slugify(question)
    q_path = os.path.join(feature_dir_, '%s.json' % q_id)
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

    answer = [normalize(a) for a in data['answer']]
    prediction = json.loads(prediction_line_)
    ranked_prediction = sorted(prediction, key=lambda k: -k['doc_score'])
    correct_rank = get_rank(prediction, answer, match_fn)
    if correct_rank > 150:
        return missing_count_, total_count_, stop_count_
    all_n_p = []
    all_n_a = []
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
        feat_file = os.path.join(feature_dir_, '%s.json' % doc_id)
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

        all_n_p.append(n_p)
        all_n_a.append(n_a)

        all_p_scores.append(doc_score)
        all_a_scores.append(ans_score)

        f_np = aggregate(all_n_p)
        f_na = aggregate(all_n_a)
        f_sp = aggregate(all_p_scores)
        f_sa = aggregate_ans(all_a_scores)

        record = OrderedDict()

        # sp, nq, np, na, ha
        record['sp'] = f_sp
        record['nq'] = list(map(float, n_q))
        record['np'] = f_np
        record['na'] = f_na
        record['sa'] = f_sa

        if i + 1 == correct_rank:
            record['stop'] = 1
            stop_count_ += 1
            write_record = True
            should_return = True
        else:
            should_return = False
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
        if should_return:
            return missing_count_, total_count_, stop_count_
    return missing_count_, total_count_, stop_count_


if __name__ == '__main__':
    # unzip trec.tgz to trec
    # below is an example run, take 114.5s(on mac mini 2012), generated 15571 records, 7291 of them are stop labels
    # python prepare_data.py -p CuratedTrec-test-lstm.preds.txt -a CuratedTrec-test.txt -f trec -r records
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_file',
                        help='prediction file, e.g. CuratedTrec-test-lstm.preds.txt')
    parser.add_argument('-a', '--answer_file', help='data set with labels, e.g. CuratedTrec-test.txt')
    parser.add_argument('-nm', '--no_multiprocess', action='store_true', help='default to use multiprocessing')
    parser.add_argument('-ns', '--negative_scale', type=int, default=10, help='scale factor for negative samples')
    parser.add_argument('-r', '--record_dir', default=None, help='dir to save generated records data set')
    parser.add_argument('-f', '--feature_dir', default=None,
                        help='dir that contains json features files, unzip squad.tgz or trec.tgz to get that dir')
    parser.add_argument('-rg', '--regex', action='store_true', help='default to use exact match')

    args = parser.parse_args()

    match_func = regex_match_score if args.regex else exact_match_score
    missing_count = 0
    total_count = 0
    stop_count = 0

    answer_file = args.answer_file
    prediction_file = args.prediction_file
    record_dir = args.record_dir
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    feature_dir = args.feature_dir
    if not os.path.exists(feature_dir):
        print('feature_dir does not exist!')
        exit(-1)
    s = time.time()
    if args.no_multiprocess:
        for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                              open(prediction_file, encoding=ENCODING)):
            missing, total, stop = process_record(data_line, prediction_line, args.negative_scale,
                                                  feature_dir, record_dir, match_func)
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
            param = (data_line, prediction_line, args.negative_scale,
                     feature_dir, record_dir, match_func)
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
