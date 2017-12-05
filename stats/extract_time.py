#!/usr/bin/env python3
import argparse
from extract_util import extract_lines

ENCODING = "utf-8"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_file', type=str, default='logs/tx2/trec_paras_100.log')

    args = parser.parse_args()

    # time_flags = ['docs retrieved', 'paragraphs predicted', 'queries processed']
    time_flags = ['docs retrieved', 'paragraphs predicted', 'queries processed']

    stage_times = [list() for _ in time_flags]

    for time_flag, stage_time in zip(time_flags, stage_times):
        for line in extract_lines(args.log_file, '%s [time]: ' % time_flag, ' s ]'):
            stage_time.append(float(line))

    for stage_time in stage_times:
        avg_time = sum(stage_time) / len(stage_time)
        print('%.4f' % avg_time)