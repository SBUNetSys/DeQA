#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from tqdm import tqdm


def load_data(filename):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_file', type=str)
    parser.add_argument('-o', '--out_file', type=str)
    args = parser.parse_args()
    with open(args.out_file, 'w', encoding='utf-8') as of:
        for n, line in tqdm(enumerate(open(args.record_file))):
            ex = json.loads(line)
            if len(ex['document']) > 400:
                print('para', n, 'too long:', len(ex['document']), 'skipped')
                continue
            ex['question_char'] = [[c for c in qw] for qw in ex['question']]
            ex['document_char'] = [[c for c in dw] for dw in ex['document']]
            of.write(json.dumps(ex) + '\n')
