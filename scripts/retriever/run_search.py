#!/usr/bin/env python3

import subprocess
import regex

# ../../data/wikipedia/galago-idx
p = subprocess.Popen(['../../data/galago/bin/galago', 'batch-search', '--index=',
                      '/Volumes/HDD/enwiki-idx-20171103', '--requested=', '5', '--casefold=true',
                      '--query=', '#combine(Anarchism)'], stdout=subprocess.PIPE)
out, err = p.communicate()
print(out.decode("utf-8").strip())


''' example results
unk-0 Q0 12_5 5 -5.86565602 galago https://en.wikipedia.org/?curid=12_5
unk-0 Q0 12_69 4 -5.85365714 galago https://en.wikipedia.org/?curid=12_69
unk-0 Q0 12_2 3 -5.85174935 galago https://en.wikipedia.org/?curid=12_2
'''

if __name__ == '__main__':
    doc_scores = []
    doc_ids = []
    doc_texts = []

    for result in out.decode("utf-8").strip().split('</TEXT>'):

        # skip <NE> field
        result = regex.sub("<NE>([^$]+)</NE>", '', result).strip()
        if not result:
            continue

        result_elements = result.split('<TEXT>')
        # print(result_elements)
        meta_info_list = result_elements[0].split()
        if len(meta_info_list) < 7:
            print('query failed')
            continue
        doc_id = meta_info_list[2]
        doc_score = meta_info_list[4]
        text = result_elements[1].strip()

        doc_ids.append(doc_id)
        doc_scores.append(doc_score)
        doc_texts.append(text)
        print('doc_id:', doc_id, 'doc_score:', doc_score)

