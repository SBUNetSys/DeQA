#!/usr/bin/env python3
import argparse
import ast
from extract_util import extract_lines

SEPARATOR_STR = 'question: '

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_file', type=str, default='logs/trec_docs_5.log')

    args = parser.parse_args()
    for i, question_line in enumerate(extract_lines(args.log_file, SEPARATOR_STR)):
            question, para_words_strings = question_line.split('paragraphs: ')
            para_words_list = ast.literal_eval(para_words_strings)
            print('%d,' % (i + 1), 'paragraphs: %d ' % len(para_words_list),
                  'words: %d ' % sum(para_words_list), 'question: %s' % question.strip())
