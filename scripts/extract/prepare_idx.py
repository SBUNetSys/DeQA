#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from tqdm import tqdm
import json
import unicodedata
import spacy
from drqa import tokenizers
from drqa.retriever import utils

nlp = spacy.load('en')


def normalize(token):
    return unicodedata.normalize('NFD', token)


def pad_array(x, max_len=None):
    """
    pad input 2d array (list of list), if max_len is given, may truncate inner 1d array:
    e.g.
    x = [[3, 46, 73, 43, 46],
         [19, 65, 64],
         [2, 82, 55, 4],
         [18, 82, 76, 18, 82, 28, 82, 27, 2, 82, 36],
         [2, 46, 82],
         [25, 65, 39],
         [55, 2]]
    pad_array(x, 5) will be:
        [[3, 46, 73, 43, 46],
         [19, 65, 64, 0, 0],
         [2, 82, 55, 4, 0],
         [18, 82, 76, 18, 82],
         [2, 46, 82, 0, 0],
         [25, 65, 39, 0, 0],
         [55, 2, 0, 0, 0]]
    while pad_array(x) will be:
        [[3, 46, 73, 43, 46, 0, 0, 0, 0, 0, 0],
         [19, 65, 64, 0, 0, 0, 0, 0, 0, 0, 0],
         [2, 82, 55, 4, 0, 0, 0, 0, 0, 0, 0],
         [18, 82, 76, 18, 82, 28, 82, 27, 2, 82, 36],
         [2, 46, 82, 0, 0, 0, 0, 0, 0, 0, 0],
         [25, 65, 39, 0, 0, 0, 0, 0, 0, 0, 0],
         [55, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    :param x: input 2d array
    :param max_len: truncate len
    :return: padded list in rectangular format
    """
    max_length = max(len(row) for row in x)
    return [row[:max_len] + [0] * ((max_len if max_len else max_length) - len(row)) for row in x]


def get_doc_indices(doc, tok2idx, char2idx):
    nlp_doc = nlp(doc)
    doc_indices = []
    char_indices = []
    for dt in nlp_doc:
        doc_indices.append(tok2idx.get(normalize(dt.text), 1))
        char_indices.append([char2idx.get(c, 1) for c in dt.text])
    return doc_indices, char_indices


def gen_query(question_):
    normalized = normalize(question_)
    tokenizer = tokenizers.get_class('simple')()
    tokens = tokenizer.tokenize(normalized)
    words = tokens.ngrams(n=1, uncased=True, filter_fn=utils.filter_ngram)
    query_ = ' '.join(words)
    return query_


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--doc_file', type=str)
    parser.add_argument('-t', '--tok2idx_file', type=str)
    parser.add_argument('-c', '--char2idx_file', type=str)
    parser.add_argument('-o', '--out_file', type=str)
    parser.add_argument('-g', '--gen_query', action='store_true')
    parser.add_argument('-pl', '--pad_char_len', type=int, default=-1)
    parser.add_argument('-ni', '--no_indent', action='store_true')

    args = parser.parse_args()

    doc_file = args.doc_file

    with open(doc_file, encoding='utf-8') as f:
        doc_data = json.load(f)

    with open(args.tok2idx_file, encoding='utf-8') as f:
        tok2idx_dict = json.load(f)

    with open(args.char2idx_file, encoding='utf-8') as f:
        char2idx_dict = json.load(f)

    data_dict = dict()
    print('processing data from %s' % args.doc_file)
    questions = []
    for k, document in tqdm(doc_data.items()):
        doc_idx, doc_char_idx = get_doc_indices(document, tok2idx_dict, char2idx_dict)
        if args.pad_char_len > 0:
            doc_char_idx = pad_array(doc_char_idx, args.pad_char_len)
        data_dict[k] = {
            'd_idx': doc_idx,
            'dc_idx': doc_char_idx
        }
        if args.gen_query:
            data_dict[k]['query'] = gen_query(document)
            data_dict[k]['text'] = document

    doc_base, doc_ext = os.path.splitext(doc_file)
    out_file = args.out_file or doc_base + '.idx' + doc_ext

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data_dict, sort_keys=True, indent=None if args.no_indent else 2))
    print('all done.')
