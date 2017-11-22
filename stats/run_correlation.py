#!/usr/bin/env python3
import json
import argparse
import ast
import collections
import os
from extract_util import extract_lines

PREDICTION_FILE = 'CuratedTrec-test-multitask-pipeline.preds'
ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_file', type=str, default='../data/datasets/CuratedTrec-test.txt')
    parser.add_argument('-l', '--log_file', type=str, default='logs/pc/trec_docs_5.log')
    parser.add_argument('-p', '--prediction_dir', type=str, default='preds/trec_docs_5')

    args = parser.parse_args()

    questions = []
    for line in open(args.answer_file, encoding=ENCODING):
        data = json.loads(line)
        question = data['question']
        questions.append(question)

    doc_ids = []
    prediction_file = os.path.join(args.prediction_dir, PREDICTION_FILE)
    for line in open(prediction_file, encoding=ENCODING):
        data = json.loads(line)
        if len(data):
            doc_id = data[0]['doc_id']
        else:
            doc_id = ''
        doc_ids.append(doc_id)

    query_doc_dict = {}
    for line in extract_lines(args.log_file, 'question_d:', ' ]'):
        question, sec_strings = line.split(', query:')
        start_index = sec_strings.index('doc_ids:') + len('doc_ids:')
        end_index = sec_strings.index(', doc_scores:')
        doc_id_strings = sec_strings[start_index:end_index]
        top_doc_ids = ast.literal_eval(doc_id_strings)
        query_doc_dict[question.strip()] = top_doc_ids

    ranks = []
    for question, doc_id in zip(questions, doc_ids):
        top_doc_ids = query_doc_dict[question]
        index = top_doc_ids.index(doc_id) + 1
        # print(question, doc_id, index)
        ranks.append(index)

    # print(ranks)
    rank_counter = collections.Counter(ranks)
    for rank in sorted(rank_counter.keys()):
        print(rank, rank_counter.get(rank))

