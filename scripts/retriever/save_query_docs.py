#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from tqdm import tqdm
from drqa.retriever import LuceneRanker

logger = logging.getLogger(__name__)
try:
    import ujson as json
except ImportError:
    import json

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question_file', type=str)
    args = parser.parse_args()
    ranker = LuceneRanker()
    question_file = args.question_file
    q_doc_dict = dict()
    id_doc_dict = dict()
    logger.info('loading queries from %s' % question_file)
    questions = []
    for line in tqdm(open(question_file)):
        data = json.loads(line)
        question = data['question']
        questions.append(question)

    logger.info('processing %d queries' % len(questions))
    ranked = ranker.batch_closest_docs(questions, k=150)

    logger.info('docs retrieved')
    all_doc_ids, all_doc_scores, all_doc_texts = zip(*ranked)
    for no, question in enumerate(tqdm(questions)):
        ids, scores, texts = all_doc_ids[no], all_doc_scores[no], all_doc_texts[no]
        scores = map(float, scores)
        id_doc_dict.update(dict(zip(ids, texts)))
        q_doc_dict[question] = (ids, scores)
        # ig = operator.itemgetter(1)
        # q_doc_dict[question] = [(doc_id, doc_score) for doc_id, doc_score in sorted(zip(ids, scores), key=ig)]
    logger.info('processed {} questions, found {} docs'.format(len(questions), len(id_doc_dict)))
    question_file_prefix = os.path.splitext(question_file)[0]

    with open(question_file_prefix + '.docs.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(id_doc_dict, sort_keys=True, indent=2))
    with open(question_file_prefix + '.questions.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(q_doc_dict, sort_keys=True, indent=2))
    logger.info('all done.')
