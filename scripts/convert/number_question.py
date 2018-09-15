#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--question_file', type=str)
    parser.add_argument('-o', '--out_file', type=str)
    args = parser.parse_args()

    question_file = args.question_file

    q_dict = dict()
    for n, line in tqdm(enumerate(open(question_file), 1)):
        data = json.loads(line)
        question = data['question']
        q_dict[n] = question

    doc_base, doc_ext = os.path.splitext(question_file)
    out_file = args.out_file or doc_base + '.questions' + doc_ext

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(q_dict, sort_keys=True, indent=2))
    print('all done.')
