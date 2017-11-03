#!/usr/bin/env python3
import argparse
import json
import os
import random


def prettify_json(f):
    if not f.endswith(".json"):
        return
    print("prettifying : {}".format(f))
    parsed = json.load(open(f, 'r'))
    pretty_path = "{}.txt".format(f)

    with open(pretty_path, 'w') as p:
        p.write(json.dumps(parsed, indent=2))
        print("saved to : {}\n".format(pretty_path))


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

    random.seed(seed)
    selected_indices = random.sample(range(0, question_size), size)
    print("selected_indices: {}".format(selected_indices))

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
    prettify_json(output_path)
    return output_path


def print_data_stats(input_json):
    if not input_json.endswith(".json"):
        print("{} is not json file".format(input_json))
        return
    json_data = json.load(open(input_json, 'r'))

    article_size = len(json_data['data'])
    paragraph_size = 0
    question_size = 0
    context_string_lengths = []
    question_string_lengths = []
    for article_index, article in enumerate(json_data['data']):
        paragraph_size += len(article["paragraphs"])
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            context_string_lengths.append(len(paragraph["context"]))
            question_size += len(paragraph["qas"])
            for qa in paragraph["qas"]:
                question_string_lengths.append(len(qa["question"]))

    print("total articles: {}".format(article_size))
    print("total paragraphs: {}".format(paragraph_size))
    print("total questions: {}".format(question_size))
    print("context_string_lengths (sorted:{}".format(sorted(context_string_lengths)))
    print("question_string_lengths (sorted): {}".format(sorted(question_string_lengths)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--print_data_stats")
    parser.add_argument('-e', "--extract_squad", nargs='*')
    parser.add_argument('-p', "--prettify_json", nargs='*')

    args = parser.parse_args()
    if args.print_data_stats:
        print_data_stats(args.print_data_stats)

    if args.extract_squad:
        input_squad_path = args.extract_squad[0]
        extract_size = int(args.extract_squad[1])
        if len(args.extract_squad) == 3:
            extracted_path = extract_squad(input_squad_path, extract_size, int(args.extract_squad[2]))
        else:
            extracted_path = extract_squad(input_squad_path, extract_size)
        if args.prettify_json:
            prettify_json(extracted_path)
    elif args.prettify_json:
        for j in args.prettify_json:
            prettify_json(j)