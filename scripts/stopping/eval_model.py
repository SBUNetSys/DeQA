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
import numpy

ENCODING = "utf-8"
DOC_MEAN = 8.5142
DOC_STD = 2.8324
I_STD = 28.56
I_MEAN = 14.08
# Z_STD = 54659
# Z_MEAN = 669.91
Z_STD = 241297
Z_MEAN = 3164
# ANS_MEAN=86486
# ANS_STD=256258
ANS_MEAN = 11588614
ANS_STD = 98865053


# def batch_predict(data_line_, prediction_line_, model, feature_dir_, match_fn_):
#     print("DEFUNCT BATCH_PREDICT")
#     data = json.loads(data_line_)
#     question = data['question']
#     q_id = slugify(question)
#     q_path = os.path.join(feature_dir_, '%s.json' % q_id)
#     n_q = [0 for _ in Tokenizer.FEAT]
#     if os.path.exists(q_path):
#         q_data = open(q_path, encoding=ENCODING).read()
#         record = json.loads(q_data)
#         q_ner = record['ner']
#         q_pos = record['pos']
#         for feat in q_ner + q_pos:
#             n_q[Tokenizer.FEAT_DICT[feat]] += 1
#
#     answer = [normalize(a) for a in data['answer']]
#     prediction = json.loads(prediction_line_)
#     #    ranked_prediction = sorted(prediction, key=lambda k: k['doc_score'])
#     ranked_prediction = sorted(prediction, key=lambda k: k['doc_score'], reverse=True)
#     correct_rank = get_rank(ranked_prediction, answer, match_fn_)
#     total_count_ = 0
#     correct_count_ = 0
#
#     if correct_rank > 150:
#         print("BAD")
#         return 0, 0
#     all_n_p = []
#     all_n_a = []
#
#     all_p_scores = []
#     all_a_scores = []
#     for i, entry in enumerate(ranked_prediction):
#
#         if i + 1 > correct_rank:
#             break
#
#         doc_id = entry['doc_id']
#         start = int(entry['start'])
#         end = int(entry['end'])
#         doc_score = entry['doc_score']
#         ans_score = entry['span_score']
#
#         p_pos = dict()
#         p_ner = dict()
#         feat_file = os.path.join(feature_dir_, '%s.json' % doc_id)
#         if os.path.exists(feat_file):
#             record = json.load(open(feat_file))
#             p_ner[doc_id] = record['ner']
#             p_pos[doc_id] = record['pos']
#         n_p = [0 for _ in Tokenizer.FEAT]
#         n_a = [0 for _ in Tokenizer.FEAT]
#         for feat in p_ner[doc_id] + p_pos[doc_id]:
#             n_p[Tokenizer.FEAT_DICT[feat]] += 1
#         for feat in p_ner[doc_id][start:end + 1] + p_pos[doc_id][start:end + 1]:
#             n_a[Tokenizer.FEAT_DICT[feat]] += 1
#
#         ################Calculate sample z score (t statistic) for answer score
#         if all_a_scores == [] or len(all_a_scores) == 1:  # dont use a_zscore feature at the beginning
#             a_zscore = 0
#         else:
#             #            sample_mean = numpy.mean(all_a_scores + [ans_score])
#             sample_mean = numpy.mean(all_a_scores)
#             #            sample_std = numpy.std(all_a_scores + [ans_score])
#             sample_std = numpy.std(all_a_scores)
#             a_zscore = (ans_score - sample_mean) / sample_std
#
#         corr_doc_score = (doc_score - DOC_MEAN) / DOC_STD
#         a_zscore_t = torch.FloatTensor(list([a_zscore]))  # 1
#
#         corr_doc_score_t = torch.FloatTensor(list([corr_doc_score]))  # 1
#         ###############
#
#         all_n_p.append(n_p)
#         all_n_a.append(n_a)
#
#         all_p_scores.append(doc_score)
#         all_a_scores.append(ans_score)
#
#         f_np = aggregate(all_n_p)
#         f_na = aggregate(all_n_a)
#         f_sp = aggregate(all_p_scores)
#         f_sa = aggregate_ans(all_a_scores)
#
#         # sp, nq, np, na, ha
#         sp = torch.FloatTensor(f_sp)  # 4x1
#         sa = torch.FloatTensor(f_sa)  # 2x1
#
#         i_ft = torch.FloatTensor([i])
#         i_std = (i - I_MEAN) / I_STD
#         i_std = torch.FloatTensor([i_std])
#
#         # OLD ONES NO GOOD
#         # np = torch.FloatTensor(list(map(float, n_q)))  # 4x58
#         # na = torch.FloatTensor(f_np)  # 4x58
#         # nq = torch.FloatTensor(f_na)  # 1x58
#
#         np = torch.FloatTensor(f_np)
#
#         # inputs = torch.cat([sp, sa, nq, np, na])
#         # Uncomment this one
#         #        inputs = torch.cat([sp, nq, np, na, a_zscore_t])
#         # inputs = torch.cat([sp, a_zscore_t])
#         # inputs = torch.cat([sp, a_zscore_t])
#
#         #        inputs = torch.cat([corr_doc_score_t, a_zscore_t, i_ft])
#         inputs = torch.cat([corr_doc_score_t, a_zscore_t])
#
#         prob = model.predict(inputs, prob=True)
#         print("Prob of STOP = {}, Correct Rank = {}, i = {}".format(prob, correct_rank, i))
#         if prob > 0.5:
#             #        if prob > 0.75:
#             if i + 1 >= correct_rank:
#                 correct_count_ += 1
#             break
#     total_count_ += 1
#     return correct_count_, total_count_


def batch_predict_test(data_line_, prediction_line_, model, feature_dir_, match_fn_, stop_at=-1):
    data = json.loads(data_line_)
    # question = data['question']
    # q_id = slugify(question)
    # q_path = os.path.join(feature_dir_, '%s.json' % q_id)
    # n_q = [0 for _ in Tokenizer.FEAT]
    # if os.path.exists(q_path):
    #     q_data = open(q_path, encoding=ENCODING).read()
    #     record = json.loads(q_data)
    #     q_ner = record['ner']
    #     q_pos = record['pos']
    #     for feat in q_ner + q_pos:
    #         n_q[Tokenizer.FEAT_DICT[feat]] += 1

    answer = [normalize(a) for a in data['answer']]
    prediction = json.loads(prediction_line_)
    ranked_prediction = sorted(prediction, key=lambda k: k['doc_score'], reverse=True)
    correct_rank = get_rank(ranked_prediction, answer, match_fn_)
    total_count_ = 0
    correct_count_ = 0

    if correct_rank > 150:
        print("BAD")
        return 0, 0, 0, ranked_prediction
    # all_n_p = []
    # all_n_a = []

    all_p_scores = []
    all_a_scores = []
    all_probs = []
    diff = 0
    repeats = 0
    all_spans = []

    es_preds = []
    stop_loc = 0
    for i, entry in enumerate(ranked_prediction):
        es_preds.append(entry)
        # doc_id = entry['doc_id']
        # start = int(entry['start'])
        # end = int(entry['end'])
        doc_score = entry['doc_score']
        ans_score = entry['span_score']
        prob = entry['prob']
        span = entry['span']

        if span in all_spans:
            repeats += 1

        all_spans.append(span)
        all_probs.append(prob)

        #        print("Threshold 1000000")
        #        ans_score=min(ans_score, 1000000) #restrict to max of million

        # p_pos = dict()
        # p_ner = dict()
        # feat_file = os.path.join(feature_dir_, '%s.json' % doc_id)
        # if os.path.exists(feat_file):
        #     record = json.load(open(feat_file))
        #     p_ner[doc_id] = record['ner']
        #     p_pos[doc_id] = record['pos']
        # n_p = [0 for _ in Tokenizer.FEAT]
        # n_a = [0 for _ in Tokenizer.FEAT]
        # for feat in p_ner[doc_id] + p_pos[doc_id]:
        #     n_p[Tokenizer.FEAT_DICT[feat]] += 1
        # for feat in p_ner[doc_id][start:end + 1] + p_pos[doc_id][start:end + 1]:
        #     n_a[Tokenizer.FEAT_DICT[feat]] += 1

        ################Calculate sample z score (t statistic) for answer score
        if all_a_scores == [] or len(all_a_scores) == 1:  # dont use a_zscore feature at the beginning
            a_zscore = 0
        else:
            #            sample_mean = numpy.mean(all_a_scores + [ans_score])
            sample_mean = numpy.mean(all_a_scores)
            #            sample_std = numpy.std(all_a_scores + [ans_score])
            sample_std = numpy.std(all_a_scores)
            #            if sample_std != 0:
            a_zscore = (ans_score - sample_mean) / sample_std
        #            else:
        #                a_zscore = 0

        # if a_zscore != 0:
        #     az_norm = (a_zscore - Z_MEAN) / Z_STD
        # else:
        #     az_norm = 0

        # a_zscore_norm = torch.FloatTensor(list([az_norm]))  # 1

        corr_doc_score = (doc_score - DOC_MEAN) / DOC_STD
        # ans_avg = (numpy.mean(all_a_scores + [ans_score]) - ANS_MEAN) / ANS_STD
        a_zscore_t = torch.FloatTensor(list([a_zscore]))  # 1
        # ans_avg = torch.FloatTensor(list([ans_avg]))  # 1

        corr_doc_score_t = torch.FloatTensor(list([corr_doc_score]))  # 1

        # prob_avg = sum(all_probs) / len(all_probs)
        # prob_avg = torch.FloatTensor([prob_avg])

        # repeats_t = torch.FloatTensor([repeats])
        ###############

        # all_n_p.append(n_p)
        # all_n_a.append(n_a)

        all_p_scores.append(doc_score)
        all_a_scores.append(ans_score)

        # f_np = aggregate(all_n_p)
        # f_na = aggregate(all_n_a)
        # f_sp = aggregate(all_p_scores)
        # f_sa = aggregate_ans(all_a_scores)

        # sp, nq, np, na, ha
        # sp = torch.FloatTensor(f_sp)  # 4x1
        # sa = torch.FloatTensor(f_sa)  # 2x1

        # i_ft = torch.FloatTensor([i])
        # i_std = (i - I_MEAN) / I_STD
        # i_std = torch.FloatTensor([i_std])

        # OLD ONES NO GOOD
        # np = torch.FloatTensor(list(map(float, n_q)))  # 4x58
        # na = torch.FloatTensor(f_np)  # 4x58
        # nq = torch.FloatTensor(f_na)  # 1x58

        # na = torch.FloatTensor(f_na)  # 4x58
        # np = torch.FloatTensor(f_np)
        # nq = torch.FloatTensor(list(map(float, n_q)))  # 4x58

        # inputs = torch.cat([sp, sa, nq, np, na])
        # Uncomment this one
        #        inputs = torch.cat([sp, nq, np, na, a_zscore_t])
        # inputs = torch.cat([sp, a_zscore_t])
        # inputs = torch.cat([sp, a_zscore_t])

        #        inputs = torch.cat([corr_doc_score_t, a_zscore_t, i_ft])
        inputs = torch.cat([corr_doc_score_t, a_zscore_t])

        prob = model.predict(inputs, prob=True)
        #        print(list(model.network.parameters()))
        if stop_at <= 0:
            print("Prob of STOP = {}, Correct Rank = {}, i = {}, answer_score = {}, REPEATS = {}".format(prob,
                                                                                                         correct_rank,
                                                                                                         i, ans_score,
                                                                                                         repeats))
            #    if prob > 0.5:
            if prob > 0.95:
                if i + 1 >= correct_rank:
                    correct_count_ += 1
                    diff = i + 1 - correct_rank
                    print("stop_at <=0 prob > 0.45 CORRECT")
                print("AVG ANS SCORE {}".format(numpy.mean(all_probs)))

                print("STD ANS SCORE {}".format(numpy.std(all_probs)))
                stop_loc = i + 1
                break
            elif i + 1 >= 40:
                print("AVG ANS SCORE {}".format(numpy.mean(all_probs)))

                print("STD ANS SCORE {}".format(numpy.std(all_probs)))

                if i + 1 >= correct_rank:
                    correct_count_ += 1
                    print("stop_at <=0 prob <= 0.45 CORRECT")
                    diff = i + 1 - correct_rank
                stop_loc = i + 1
                break
        else:
            if i + 1 == stop_at:
                #        if prob > 0.75:
                if i + 1 >= correct_rank:
                    correct_count_ += 1
                    diff = i + 1 - correct_rank
                    print("stop_at > 0, CORRECT")
                stop_loc = i + 1
                break

    print("stop at: ", stop_loc)
    assert stop_loc == len(es_preds)
    total_count_ += 1
    return correct_count_, total_count_, diff, es_preds


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
    parser.add_argument('--stop_at', default=-1, type=int)

    args = parser.parse_args()

    match_func = regex_match_score if args.regex else exact_match_score

    answer_file = args.answer_file
    prediction_file = args.prediction_file

    diffs = []

    feature_dir = args.feature_dir
    # if not os.path.exists(feature_dir):
    #     print('feature_dir does not exist!')
    #     exit(-1)
    s = time.time()
    eval_model = EarlyStoppingModel.load(args.model_file)
    eval_model.network.cpu()
    total_count = 0
    correct_count = 0

    #    print('using multiprocessing...')
    result_handles = []
    #    async_pool = ProcessPool()

    for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                          open(prediction_file, encoding=ENCODING)):
        param = (data_line, prediction_line, eval_model, feature_dir, match_func, args.stop_at)
        #  handle = async_pool.apply_async(batch_predict, param)
        handle = batch_predict_test(*param)
        result_handles.append(handle)

    with open(prediction_file + '.es.txt', 'w') as f:
        for result in result_handles:
            #        correct, total = result.get()
            correct, total, dif, es_prediction = result
            f.write(json.dumps(es_prediction) + '\n')
            correct_count += correct
            total_count += total
            if total > 0:
                diffs.append(dif)
    #        if total_count % 100 ==0:
    #            print('processed %d/%d, %2.4f' % (correct_count, total_count, correct_count / total_count))
    #        sys.stdout.flush()

    e = time.time()
    print('correct_count:', correct_count, 'total_count:', total_count, 'acc:', correct_count / total_count)
    print('Diff Mean: ', numpy.mean(diffs), 'diff std:', numpy.std(diffs))
    print('took %.4f s' % (e - s))
