#!/usr/bin/env python3

END_STR = ' ]'


def extract_lines(log_file_, flag_words):
    lines_ = []
    for line_ in open(log_file_, encoding="utf-8"):
        if flag_words in line_:
            target_line = line_[line_.index(flag_words) + len(flag_words):line_.index(END_STR)]
            lines_.append(target_line)
    return lines_
