import json
import argparse
import os
from drqa import tokenizers
from drqa.retriever import utils
from drqa.reader.utils import slugify


def gen_query(question_):
    normalized = utils.normalize(question_)
    tokenizer = tokenizers.get_class('simple')()
    tokens = tokenizer.tokenize(normalized)
    words = tokens.ngrams(n=1, uncased=True, filter_fn=utils.filter_ngram)
    query_ = ' '.join(words)
    return query_


def gen_name(question_):
    return slugify(question_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str)

    args = parser.parse_args()
    data_file = args.data_file
    data_file_name, ext = os.path.splitext(data_file)
    question_file = data_file_name + '.question' + ext
    query_file = data_file_name + '.query' + ext
    name_file = data_file_name + '.name' + ext

    with open(question_file, 'w') as fq, open(query_file, 'w') as fy, open(name_file, 'w') as fn:
        for data_line in open(data_file):
            data = json.loads(data_line)
            question = data['question']
            fq.write(question + '\n')
            query = gen_query(question)
            fy.write(query + '\n')
            name = gen_name(question)
            fn.write(name + '\n')

