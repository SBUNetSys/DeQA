#!/usr/bin/env python3
import argparse
import json
import os
import random

ENCODING = "utf-8"


def extract_squad(input_json, size=100, seed=0):
    if not input_json.endswith(".json"):
        print("{} is not json file".format(input_json))
        return
    json_data = json.load(open(input_json, 'r'))

    article_size = len(json_data['data'])
    paragraph_size = 0
    question_size = 0
    for article_index, article in enumerate(json_data['data']):
        paragraph_size += len(article["paragraphs"])
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            question_size += len(paragraph["qas"])
    print("total articles: {}".format(article_size))
    print("total paragraphs: {}".format(paragraph_size))
    print("total questions: {}".format(question_size))

    question_index = 0
    selected_data = {}
    articles = []
    for article_index, article in enumerate(json_data['data']):
        # print("article index:{}, title:{}".format(article_index, article["title"]))
        # print("    paragraphs: {}".format(len(article["paragraphs"])))
        paragraph_size += len(article["paragraphs"])
        paragraphs = []
        append_article = False
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            # print("paragraph index:{}".format(paragraph_index))
            # print("       questions: {}".format(len(paragraph["qas"])))
            question_size += len(paragraph["qas"])
            qas = []
            append_paragraph = False
            for qa_index, qa in enumerate(paragraph["qas"]):
                if question_index in selected_indices:
                    # this is the question we want
                    append_article = True
                    append_paragraph = True
                    qas.append(qa)
                question_index += 1
            if append_paragraph:
                paragraph["qas"] = qas
                paragraphs.append(paragraph)
        if append_article:
            article["paragraphs"] = paragraphs
            articles.append(article)
    selected_data["data"] = articles
    selected_data["version"] = "1.1"
    output_dir = os.path.dirname(input_json)
    output_name = "{}-{}.json".format(os.path.splitext(os.path.basename(input_json))[0], size)
    output_path = os.path.join(output_dir, output_name)
    json.dump(selected_data, open(output_path, 'w'))
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_file',
                        default='../../data/datasets/SQuAD-v1.1-dev-multitask-pipeline.preds')
    parser.add_argument('-a', '--answer_file', default='../../data/datasets/SQuAD-v1.1-dev.txt')
    parser.add_argument('-s', '--size', type=int, default=100, help='sample size')
    parser.add_argument('-rs', '--random_seed', type=int, default=0, help='random seed for reproducibility')

    args = parser.parse_args()
    sample_size = args.size
    if not sample_size:
        print('no sample size given')
        exit(-1)
    random.seed(args.random_seed)
    answer_file = args.answer_file
    prediction_file = args.prediction_file
    data_pairs = [(d_l, p_l) for d_l, p_l in
                  zip(open(answer_file, encoding=ENCODING),
                      open(prediction_file, encoding=ENCODING))]

    total_samples = len(data_pairs)
    selected_indices = random.sample(range(0, total_samples), sample_size)
    # print("selected_indices: {}".format(selected_indices))
    a_base, a_ext = os.path.splitext(answer_file)
    a_out_file = "{}-{}{}".format(a_base, sample_size, a_ext)

    p_base, p_ext = os.path.splitext(prediction_file)
    p_out_file = "{}-{}{}".format(p_base, sample_size, p_ext)
    with open(a_out_file, 'w') as fa, open(p_out_file, 'w') as fp:
        for idx, pair in enumerate(data_pairs):
            if idx in selected_indices:
                answer_data, prediction_data = pair
                fa.write(answer_data)
                fp.write(prediction_data)
