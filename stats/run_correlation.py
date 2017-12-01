#!/usr/bin/env python3
import json
import argparse
import ast
import collections
import os
from extract_util import extract_lines
from drqa.retriever.utils import normalize
from drqa.reader.utils import regex_match_score, metric_max_over_ground_truths


PREDICTION_FILE = 'CuratedTrec-test-multitask-pipeline.preds'
ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_file', type=str, default='../data/datasets/CuratedTrec-test.txt')
    parser.add_argument('-l', '--log_file', type=str, default='logs/pc/trec_docs_5.log')
    parser.add_argument('-p', '--prediction_dir', type=str, default='preds/trec_docs_5')

    args = parser.parse_args()

    questions = []
    answers = []
    for line in open(args.answer_file, encoding=ENCODING):
        data = json.loads(line)
        question = data['question']
        questions.append(question)
        answer = [normalize(a) for a in data['answer']]
        answers.append(answer)

    doc_ids_ans = []
    prediction_file = os.path.join(args.prediction_dir, PREDICTION_FILE)
    for line, answer in zip(open(prediction_file, encoding=ENCODING), answers):
        data = json.loads(line)
        if len(data):
            prediction = normalize(data[0]['span'])
            id_ans = data[0]['doc_id']
        else:
            prediction = ''
            id_ans = ''
        exact_match = metric_max_over_ground_truths(regex_match_score, prediction, answer)
        doc_ids_ans.append((id_ans, exact_match))

    query_doc_dict = {}
    for line in extract_lines(args.log_file, 'question_d:', ' ]'):
        question, sec_strings = line.split(', query:')
        start_index = sec_strings.index('doc_ids:') + len('doc_ids:')
        end_index = sec_strings.index(', doc_scores:')
        doc_id_strings = sec_strings[start_index:end_index]
        top_doc_ids = ast.literal_eval(doc_id_strings)
        query_doc_dict[question.strip()] = top_doc_ids

    ranks = []
    right_ranks = []
    for question, id_ans in zip(questions, doc_ids_ans):
        doc_id, ans = id_ans
        top_doc_ids = query_doc_dict[question]
        index = top_doc_ids.index(doc_id) + 1
        # print(question, doc_id, index)
        ranks.append(index)
        if ans:
            right_ranks.append(index)

    rank_counter = collections.Counter(ranks)
    acc_rank = 0
    for rank in sorted(rank_counter.keys()):
        num = rank_counter.get(rank)
        rate = num / len(ranks) * 100
        acc_rank += num
        acc_rate = acc_rank / len(ranks) * 100
        print('%s, %s, %.1f%%, %.1f%%' % (rank, rank_counter.get(rank), rate, acc_rate))

    print()
    acc_rank = 0
    right_rank_counter = collections.Counter(right_ranks)
    for rank in sorted(right_rank_counter.keys()):
        num = right_rank_counter.get(rank)
        rate = num / len(right_ranks) * 100
        acc_rank += num
        acc_rate = acc_rank / len(right_ranks) * 100
        print('%s, %s, %.1f%%, %.1f%%' % (rank, right_rank_counter.get(rank), rate, acc_rate))

