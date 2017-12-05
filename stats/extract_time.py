#!/usr/bin/env python3
import argparse
import ast
from extract_util import extract_lines

ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_file', type=str, default='logs/trec_paras_150.log')

    args = parser.parse_args()

    # time_flags = ['docs retrieved', 'paragraphs predicted', 'queries processed']
    time_flags = ['docs retrieved', 'paragraphs predicted', 'queries processed',
                  'vectorize', 'batchify', 'input processing',
                  'embedding lookup', 'weighted question attention emb', 'doc_rnn',
                  'question_rnn', 'question_self_attn', 'question_weighted_avg',
                  'start_attn', 'end_attn', 'answer decoding']

    stage_times = [list() for _ in time_flags]

    for time_flag, stage_time in zip(time_flags, stage_times):
        for line in extract_lines(args.log_file, '%s [time]: ' % time_flag, ' s ]'):
            stage_time.append(float(line))

    for name, stage_time in zip(time_flags[:3], stage_times[:3]):
        avg_time = sum(stage_time) / len(stage_time)
        print('%s, %.4f' % (name, avg_time))
    for name, stage_time in zip(time_flags[3:], stage_times[3:]):
        if stage_time:
            avg_time = sum(stage_time) * 150 / len(stage_time)
            print('%s, %.4f' % (name, avg_time))

    # query_doc_dict = {}
    # for line in extract_lines(args.log_file, 'question_d:', ' ]'):
    #     question, sec_strings = line.split(', query:')
    #     start_index = sec_strings.index('doc_ids:') + len('doc_ids:')
    #     end_index = sec_strings.index(', doc_scores:')
    #     doc_id_strings = sec_strings[start_index:end_index]
    #     top_doc_ids = ast.literal_eval(doc_id_strings)
    #     query_doc_dict[question.strip()] = top_doc_ids
    #     if len(top_doc_ids) != 150:
    #         print(question, len(top_doc_ids))
