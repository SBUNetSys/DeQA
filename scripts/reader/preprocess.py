#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preprocess the SQuAD dataset for training."""

import argparse
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from multiprocessing import freeze_support
from multiprocessing.util import Finalize

from drqa import tokenizers

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------

TOK = None


def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'chars': tokens.chars(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
    }
    return output


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        data = json.load(f)['data']
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    for article in data:
        for paragraph in article['paragraphs']:
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                if 'answers' in qa:
                    output['answers'].append(qa['answers'])
    return output


def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert (len(start) <= 1)
    assert (len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]


def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    tokenizer_class = tokenizers.get_class(tokenizer)
    make_pool = partial(Pool, workers, initializer=init)
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
    q_tokens = workers.map(tokenize, data['questions'])
    workers.close()
    workers.join()

    workers = make_pool(
        initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}})
    )
    c_tokens = workers.map(tokenize, data['contexts'])
    workers.close()
    workers.join()

    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        question_char = q_tokens[idx]['chars']
        qlemma = q_tokens[idx]['lemma']
        document = c_tokens[data['qid2cid'][idx]]['words']
        document_char = c_tokens[data['qid2cid'][idx]]['chars']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']
        ans_tokens = []
        if len(data['answers']) > 0:
            for ans in data['answers'][idx]:
                found = find_answer(offsets,
                                    ans['answer_start'],
                                    ans['answer_start'] + len(ans['text']))
                if found:
                    ans_tokens.append(found)
        yield {
            'id': data['qids'][idx],
            'question': question,
            'question_char': question_char,
            'document': document,
            'document_char': document_char,
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
        }


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
    parser.add_argument('out_dir', type=str, help='Path to output file dir')
    parser.add_argument('--split', type=str, help='Filename for train/dev split')
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--tokenizer', type=str, default='corenlp')
    args = parser.parse_args()

    t0 = time.time()

    in_file = os.path.join(args.data_dir, args.split + '.json')
    print('Loading dataset %s' % in_file, file=sys.stderr)
    dataset = load_dataset(in_file)

    out_file = os.path.join(args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer))
    print('Will write to file %s' % out_file, file=sys.stderr)
    with open(out_file, 'w') as f:
        for ex in process_dataset(dataset, args.tokenizer, args.num_workers):
            f.write(json.dumps(ex) + '\n')
    print('Total time: %.4f (s)' % (time.time() - t0))
