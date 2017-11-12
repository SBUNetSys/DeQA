#!/usr/bin/env python3
import json
import argparse
import ast
import collections

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='../data/datasets/CuratedTrec-test.txt')
    parser.add_argument('--query_file', type=str, default='CuratedTrec-test-query.txt')
    parser.add_argument('--prediction_file', type=str, default='CuratedTrec-test-preds.txt')

    args = parser.parse_args()

    questions = []
    for line in open(args.data_file):
        data = json.loads(line)
        question = data['question']
        questions.append(question)

    doc_ids = []
    for line in open(args.prediction_file):
        data = json.loads(line)
        if len(data):
            doc_id = data[0]['doc_id']
        else:
            doc_id = ''
        doc_ids.append(doc_id)

    query_doc_dict = {}
    for line in open(args.query_file):
        question, doc_id_strings = line.split('docID: ')
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

