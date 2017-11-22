#!/usr/bin/env python3
import json
import argparse
import ast
import collections
import os
from extract_util import extract_lines

ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_file', type=str, default='logs/tx2/trec_paras_100.log')

    args = parser.parse_args()

    retrieval_time = []
    prediction_time = []
    total_time = []

    for line in extract_lines(args.log_file, 'docs retrieved [time]: ', ' s ]'):
        retrieval_time.append(line)

    for line in extract_lines(args.log_file, 'paragraphs predicted [time]: ', ' s ]'):
        prediction_time.append(line)

    for line in extract_lines(args.log_file, 'queries processed [time]: ', ' s ]'):
        total_time.append(line)

    for r, p, t in zip(retrieval_time, prediction_time, total_time):
        print(r, p, t, '%.4f %%' % (float(p)/float(t)*100))
