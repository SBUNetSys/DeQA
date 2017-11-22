#!/usr/bin/env python3
import json
import argparse
import ast

SEPARATOR_STR = 'question: '
END_STR = ' ]'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='logs/trec_docs_5.log')

    args = parser.parse_args()

    i = 0

    for line in open(args.log_file, encoding="utf-8"):
        if SEPARATOR_STR in line:
            question_line = line[line.index(SEPARATOR_STR)+len(SEPARATOR_STR):line.index(END_STR)]

            question, para_words_strings = question_line.split('paragraphs: ')
            para_words_list = ast.literal_eval(para_words_strings)
            print('%d,' % (i + 1), 'paragraphs: %d ' % len(para_words_list),
                  'words: %d ' % sum(para_words_list), 'question: %s' % question.strip())
            i += 1
