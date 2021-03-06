#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate the accuracy of the DrQA retriever module."""

import regex as re
import logging
import argparse
import json
import time
import os

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from drqa import retriever, tokenizers
from drqa.retriever import utils

# ------------------------------------------------------------------------------
# Multiprocessing target functions.
# ------------------------------------------------------------------------------

PROCESS_TOK = None


def init(tokenizer_class, tokenizer_opts):
    global PROCESS_TOK
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    # PROCESS_DB = db_class(**db_opts)
    # Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answer, doc_text, match):
    """Check if a document contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    global PROCESS_DB, PROCESS_TOK
    # text = PROCESS_DB.get_doc_text(doc_id)
    text = utils.normalize(doc_text)
    if match == 'string':
        # Answer is a list of possible strings
        text = PROCESS_TOK.tokenize(text).words(uncased=True)
        for single_answer in answer:
            single_answer = utils.normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    elif match == 'regex':
        # Answer is a regex
        single_answer = utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return True
    return False


def get_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answers_, (doc_ids, doc_scores, doc_texts) = answer_doc
    answers_ = list(set(answers_))  # remove duplicates
    for doc_id, doc_text in zip(doc_ids, doc_texts):
        if has_answer(answers_, doc_text, match):
            # print('answer:', answers_, 'docID:', doc_id, 1)
            return 1
    # print('answer:', answers_, 'docID:', 0, 0)
    return 0


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--ranker', type=str, default='tfidf')
    parser.add_argument('--db_path', type=str, default=None,
                        help='Path to Document DB or index')
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--sim_func', type=str, default='lm')
    parser.add_argument('--n-docs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--match', type=str, default='string',
                        choices=['regex', 'string'])
    args = parser.parse_args()

    # start time
    start = time.time()

    # read all the data and store it
    logger.info('Reading data ...')
    questions = []
    answers = []
    for line in open(args.dataset):
        data = json.loads(line)
        question = data['question']
        answer = data['answer']
        questions.append(question)
        answers.append(answer)

    # get the closest docs for each question.
    logger.info('Initializing ranker...')

    if args.ranker.lower().startswith('g'):
        ranker = retriever.get_class('galago')(use_keyword=True, index_path=args.db_path)
    elif args.ranker.lower().startswith('s'):
        ranker = retriever.get_class('sql')(db_path=args.db_path)
    elif args.ranker.lower().startswith('l'):
        ranker = retriever.get_class('lucene')(index_path=args.db_path)
    else:
        ranker = retriever.get_class('tfidf')(tfidf_path=args.model, db_path=args.db_path)

    logger.info('Ranking and retrieving...')
    closest_docs = ranker.batch_closest_docs(questions, k=args.n_docs, num_workers=args.num_workers)
    answers_docs = zip(answers, closest_docs)

    # define processes
    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}

    processes = ProcessPool(processes=args.num_workers, initializer=init, initargs=(tok_class, tok_opts))

    # compute the scores for each pair, and print the statistics
    logger.info('Computing scores...')
    get_score_partial = partial(get_score, match=args.match)
    scores = processes.map(get_score_partial, answers_docs)

    filename = os.path.basename(args.dataset)
    stats = (
        "\n" + "-" * 50 + "\n" +
        "{filename}\n" +
        "Examples:\t\t\t{total}\n" +
        "Matches in top {k}:\t\t{m}\n" +
        "Match % in top {k}:\t\t{p:2.2f}\n" +
        "Total time:\t\t\t{t:2.4f} (s)\n"
    ).format(
        filename=filename,
        total=len(scores),
        k=args.n_docs,
        m=sum(scores),
        p=(sum(scores) / len(scores) * 100),
        t=time.time() - start,
    )

    print(stats)
