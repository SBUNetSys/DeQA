#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import OrderedDict
from multiprocessing import Pool as ProcessPool

import numpy as np
import xgboost
from utils import exact_match_score, regex_match_score, get_rank
from utils import normalize, metric_max_over_ground_truths

# from utils import slugify

ENCODING = "utf-8"


def process_record(data_line_, prediction_line_, neg_gap_, match_fn):
    records_ = []
    stop_count_ = 0
    data = json.loads(data_line_)
    # question = data['question']
    # q_id = slugify(question)

    answer = [normalize(a) for a in data['answer']]
    prediction = json.loads(prediction_line_)
    # MAKE SURE REVERSE IS TRUE
    ranked_prediction = sorted(prediction, key=lambda k: k['doc_score'], reverse=True)
    correct_rank = get_rank(prediction, answer, match_fn)
    if correct_rank > 150:
        #  if correct_rank < 50 or correct_rank > 150:
        return records_, stop_count_

    all_p_scores = []
    all_a_scores = []
    all_a_zscores = []
    all_spans = []
    repeats = 0
    for i, entry in enumerate(ranked_prediction):
        # doc_id = entry['doc_id']
        # start = int(entry['start'])
        # end = int(entry['end'])
        doc_score = entry['doc_score']
        ans_score = entry['span_score']
        span = entry['span']

        if span in all_spans:
            repeats += 1

        all_spans.append(span)

        # Calculate sample z score (t statistic) for answer score
        if all_a_scores == [] or len(
                all_a_scores) == 1:  # dont use a_zscore feature at the beginning or if we only have 1
            a_zscore = 0
        else:  # Take the sample mean of the previous ones, take zscore of the current with respect to that
            #            sample_mean = np.mean(all_a_scores + [ans_score])
            sample_mean = np.mean(all_a_scores)
            #            sample_std = np.std(all_a_scores + [ans_score])
            sample_std = np.std(all_a_scores)
            if sample_std <= 0.0:
                a_zscore = 0
            else:
                a_zscore = (ans_score - sample_mean) / sample_std

        # THESE ARE FOR STATISTISTICS OVER ENTIRE DATA SET, IGNORE
        # all_doc_scores.append(doc_score)

        all_a_zscores.append(a_zscore)
        max_zscore = max(all_a_zscores)
        # corr_doc_score = (doc_score - DOC_MEAN) / DOC_STD
        # corr_ans_mean_score = (np.mean(all_a_scores + [ans_score]) - ANS_MEAN) / ANS_STD

        all_p_scores.append(doc_score)
        all_a_scores.append(ans_score)
        # corr_doc_score = (doc_score - DOC_MEAN) / DOC_STD

        record = OrderedDict()

        # record['a_zscore'] = a_zscore
        record['max_zscore'] = max_zscore
        record['ans_score'] = ans_score
        record['doc_score'] = doc_score
        repeats_2 = 1 if repeats == 2 else 0
        repeats_3 = 1 if repeats == 3 else 0
        repeats_4 = 1 if repeats == 4 else 0
        repeats_5 = 1 if repeats >= 5 else 0
        past20 = 1 if i >= 20 else 0
        # record['i'] = i
        # record['repeats_2'] = repeats_2
        # record['repeats_3'] = repeats_3
        # record['repeats_4'] = repeats_4
        # record['repeats_5'] = repeats_5
        record['past20'] = past20
        record['repeats'] = repeats
        match = metric_max_over_ground_truths(match_fn, normalize(span), answer)
        if match:
            record['stop'] = 1
            stop_count_ += 1
            records_.append(record)
            if stop_count_ >= 3:
                return records_, stop_count_
        else:
            # if i % neg_gap_ == 0:
            record['stop'] = 0
            records_.append(record)
    return records_, stop_count_


def gen_records(args):
    match_func = exact_match_score if args.no_regex else regex_match_score

    answer_file = args.answer_file
    prediction_file = args.prediction_file
    record_dir = os.path.dirname(args.record_file)
    os.makedirs(record_dir, exist_ok=True)

    total_count = 0
    stop_count = 0
    all_records = []
    if args.no_multiprocess:
        for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                              open(prediction_file, encoding=ENCODING)):
            records, stop = process_record(data_line, prediction_line, args.negative_scale, match_func)
            all_records.extend(records)
            total_count += len(records)
            stop_count += stop
            print('processed %d records, stop: %d' % (total_count, stop_count))
            sys.stdout.flush()
    else:
        print('using multiprocessing...')
        result_handles = []
        async_pool = ProcessPool()
        for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                              open(prediction_file, encoding=ENCODING)):
            param = (data_line, prediction_line, args.negative_scale, match_func)
            handle = async_pool.apply_async(process_record, param)
            result_handles.append(handle)
        for result in result_handles:
            records, stop = result.get()
            all_records.extend(records)
            total_count += len(records)
            stop_count += stop
            print('processed %d records, stop: %d' % (total_count, stop_count))
            sys.stdout.flush()
    with open(args.record_file, 'w', encoding=ENCODING) as f:
        f.write(json.dumps(all_records, indent=2))


def get_data(record_file):
    with open(record_file, encoding=ENCODING) as f:
        data = json.load(f)

    d = [list(da.values()) for da in data]
    data = np.asarray(d)
    x = data[:, :-1]
    label = data[:, -1]
    return x, label.astype(int)


def train_classifier(args):
    data_dir = os.path.dirname(args.record_file)
    x_train, label_train = get_data(args.record_file)
    x_eval, label_eval = get_data(args.test_record)
    data_train = xgboost.DMatrix(x_train, label=label_train)
    data_test = xgboost.DMatrix(x_eval, label=label_eval)
    params = {'max_depth': 6, 'eta': 0.3, 'silent': 0, 'objective': 'binary:logistic',
              'eval_metric': ['error', 'auc', 'map']}

    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    num_round = 20
    bst = xgboost.train(params, data_train, num_round, watchlist)
    model_file = args.model_file or os.path.join(data_dir, '{}.xgb'.format(args.classifier))
    model_dir = os.path.dirname(model_file)
    os.makedirs(model_dir, exist_ok=True)
    bst.save_model(model_file)
    xgboost.plot_importance(bst)
    xgboost.plot_tree(bst)


def eval_end2end(args):
    out_file = args.out_file
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    prediction_file = args.prediction_file
    data_dir = os.path.dirname(prediction_file)

    model_file = args.model_file or os.path.join(data_dir, '{}.xgb'.format(args.classifier))
    bst = xgboost.Booster()
    bst.load_model(model_file)

    stop_count = 0
    processed = 0
    with open(out_file, 'w', encoding=ENCODING) as of:
        for prediction_line in open(prediction_file, encoding=ENCODING):

            out_predictions = []

            all_spans = []
            all_a_scores = []
            all_a_zscores = []
            repeats = 0

            prediction = json.loads(prediction_line)

            for i, entry in enumerate(sorted(prediction, key=lambda k: k['doc_score'], reverse=True)):
                out_predictions.append(entry)
                # doc_id = entry['doc_id']
                # start = int(entry['start'])
                # end = int(entry['end'])
                doc_score = entry['doc_score']
                ans_score = entry['span_score']
                span = entry['span']

                if span in all_spans:
                    repeats += 1

                all_spans.append(span)

                # Calculate sample z score (t statistic) for answer score
                if all_a_scores == [] or len(
                        all_a_scores) == 1:  # dont use a_zscore feature at the beginning or if we only have 1
                    a_zscore = 0
                else:  # Take the sample mean of the previous ones, take zscore of the current with respect to that
                    #            sample_mean = np.mean(all_a_scores + [ans_score])
                    sample_mean = np.mean(all_a_scores)
                    #            sample_std = np.std(all_a_scores + [ans_score])
                    sample_std = np.std(all_a_scores)
                    if sample_std <= 0.0:
                        a_zscore = 0
                    else:
                        a_zscore = (ans_score - sample_mean) / sample_std

                all_a_zscores.append(a_zscore)
                max_zscore = max(all_a_zscores)

                repeats_2 = 1 if repeats == 2 else 0
                repeats_3 = 1 if repeats == 3 else 0
                repeats_4 = 1 if repeats == 4 else 0
                repeats_5 = 1 if repeats >= 5 else 0
                past20 = 1 if i >= 20 else 0
                x = [max_zscore, ans_score, doc_score, past20, repeats]
                feature_x = np.reshape(x, (1, -1))
                feature_mat = xgboost.DMatrix(feature_x)
                stop_prob = bst.predict(feature_mat)

                if stop_prob > args.stop_threshold:
                    stop_count += 1
                    print(stop_prob, 'stopped at:', i + 1, stop_count, processed)
                    break

            processed += 1
            of.write(json.dumps(out_predictions) + '\n')
            print('processed', stop_count, processed)


if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)

    subparsers = parent_parser.add_subparsers(help='commands')

    gen_parser = subparsers.add_parser('gen', parents=[parent_parser], help='generate records')
    gen_parser.add_argument('-rf', '--record_file', default=None, help='file to save generated records')
    gen_parser.add_argument('-p', '--prediction_file', help='prediction file, e.g. CuratedTrec-test.preds.txt')
    gen_parser.add_argument('-a', '--answer_file', help='data set with labels, e.g. CuratedTrec-test.txt')
    gen_parser.add_argument('-nr', '--no_regex', action='store_true', help='default to use regex match')
    gen_parser.add_argument('-nm', '--no_multiprocess', action='store_true', help='default to use multiprocessing')
    gen_parser.add_argument('-ns', '--negative_scale', type=int, default=10, help='scale factor for negative samples')

    classifier_parser = subparsers.add_parser('train', parents=[parent_parser], help='train classifier')

    classifier_parser.add_argument('-c', '--classifier', default='logistic', choices=['linear', 'logistic'],
                                   help='classifier for xgboost')
    classifier_parser.add_argument('-mf', '--model_file', default=None, help='stopping model')
    classifier_parser.add_argument('-rf', '--record_file', default=None, help='train records')
    classifier_parser.add_argument('-tr', '--test_record', type=str, help='record file for testing')
    classifier_parser.add_argument('-st', '--stop_threshold', default=0.5, type=float)

    eval_parser = subparsers.add_parser('eval', parents=[parent_parser], help='eval end to end accuracy')

    eval_parser.add_argument('-c', '--classifier', default='logistic', choices=['linear', 'logistic'],
                             help='classifier for xgboost')
    eval_parser.add_argument('-p', '--prediction_file', help='prediction file, e.g. CuratedTrec-test.preds.txt')
    eval_parser.add_argument('-o', '--out_file', help='data set with labels, e.g. CuratedTrec-test.txt')
    eval_parser.add_argument('-mf', '--model_file', default=None, help='stopping model')
    eval_parser.add_argument('-nr', '--no_regex', action='store_true', help='default to use regex match')
    eval_parser.add_argument('-st', '--stop_threshold', default=0.5, type=float)

    parent_args = parent_parser.parse_args()
    # print(parent_parser.parse_args())

    if parent_args == gen_parser.parse_args():
        # print(gen_parser.parse_args())
        gen_records(gen_parser.parse_args())

    if parent_args == classifier_parser.parse_args():
        # print(classifier_parser.parse_args())
        train_classifier(classifier_parser.parse_args())

    if parent_args == eval_parser.parse_args():
        # print(eval_parser.parse_args())
        eval_end2end(eval_parser.parse_args())
