#!/usr/bin/env python3
import json
import argparse
import os
from utils import normalize
from utils import exact_match_score, regex_match_score, get_rank
from utils import slugify, aggregate, aggregate_ans
from utils import Tokenizer
from StoppingModel import EarlyStoppingModel
import torch
import time
from multiprocessing import Pool as ProcessPool
import sys

ENCODING = "utf-8"


def batch_predict(data_line_, prediction_line_, model, feature_dir_, match_fn_):
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

    answer = [normalize(a) for a in data['answer']]
    prediction = json.loads(prediction_line_)
    ranked_prediction = sorted(prediction, key=lambda k: k['doc_score'])
    correct_rank = get_rank(ranked_prediction, answer, match_fn_)
    total_count_ = 0
    correct_count_ = 0

    if correct_rank > 150:
        return 0, 0
    all_n_p = []
    all_n_a = []

    all_p_scores = []
    all_a_scores = []

    for i, entry in enumerate(ranked_prediction):

        if i + 1 > correct_rank:
            break

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
        # f_sa = aggregate_ans(all_a_scores)

        # sp, nq, np, na, ha
        sp = torch.FloatTensor(f_sp)  # 4x1
        # sa = torch.FloatTensor(f_sa)  # 2x1

        np = torch.FloatTensor(list(map(float, n_q)))  # 4x58
        na = torch.FloatTensor(f_np)  # 4x58
        nq = torch.FloatTensor(f_na)  # 1x58
        # inputs = torch.cat([sp, sa, nq, np, na])
        inputs = torch.cat([sp, nq, np, na])
        prob = model.predict(inputs, prob=True)
        if prob > 0.5:
            if i + 1 >= correct_rank:
                correct_count_ += 1
            break
    total_count_ += 1
    return correct_count_, total_count_


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_file',
                        help='prediction file, e.g. CuratedTrec-test-lstm.preds.txt')
    parser.add_argument('-a', '--answer_file', help='data set with labels, e.g. CuratedTrec-test.txt')
    parser.add_argument('-f', '--feature_dir', default=None,
                        help='dir that contains json features files, unzip squad.tgz or trec.tgz to get that dir')
    parser.add_argument('-rg', '--regex', action='store_true', help='default to use exact match')
    parser.add_argument('-m', '--model_file', default=None, help='stopping model')
    parser.add_argument('-nm', '--no_multiprocess', action='store_true', help='default to use multiprocessing')

    args = parser.parse_args()

    match_func = regex_match_score if args.regex else exact_match_score

    answer_file = args.answer_file
    prediction_file = args.prediction_file

    feature_dir = args.feature_dir
    if not os.path.exists(feature_dir):
        print('feature_dir does not exist!')
        exit(-1)
    s = time.time()
    eval_model = EarlyStoppingModel.load(args.model_file)
    eval_model.network.cpu()
    total_count = 0
    correct_count = 0

    print('using multiprocessing...')
    result_handles = []
    async_pool = ProcessPool()

    for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                          open(prediction_file, encoding=ENCODING)):
        param = (data_line, prediction_line, eval_model, feature_dir, match_func)
        handle = async_pool.apply_async(batch_predict, param)
        result_handles.append(handle)

    for result in result_handles:
        correct, total = result.get()
        correct_count += correct
        total_count += total
        if total_count % 100 ==0:
            print('processed %d/%d, %2.4f' % (correct_count, total_count, correct_count / total_count))
        sys.stdout.flush()

    e = time.time()
    print('correct_count:', correct_count, 'total_count:', total_count, 'acc:', correct_count / total_count)
    print('took %.4f s' % (e - s))
