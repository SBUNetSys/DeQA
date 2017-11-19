#!/usr/bin/env python3
import json
import argparse
import ast
import collections

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--para_file', type=str, default='CuratedTrec-test-para.txt')

    args = parser.parse_args()

    for i, line in enumerate(open(args.para_file)):
        question, para_words_strings = line.split('paragraphs: ')
        para_words_list = ast.literal_eval(para_words_strings)
        print('%d,' % (i + 1), 'paragraphs: %d ' % len(para_words_list),
              'words: %d ' % sum(para_words_list), 'question: %s' % question.strip())
