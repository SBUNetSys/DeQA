#!/usr/bin/env python3


def extract_lines(log_file_, flag_words, end_flag=None):
    lines_ = []
    for line_ in open(log_file_, encoding="utf-8"):
        if flag_words in line_:
            end_index = line_.index(end_flag) if end_flag else None
            target_line = line_[line_.index(flag_words) + len(flag_words):end_index]
            lines_.append(target_line)
    return lines_
