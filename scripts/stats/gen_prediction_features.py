#!/usr/bin/env python3
import argparse
import os
import spacy
import ujson as json
from drqa.retriever.utils import normalize
from drqa.reader.utils import exact_match_score, regex_match_score, metric_max_over_ground_truths

ENCODING = "utf-8"
nlp = spacy.load('en')


def get_rank(prediction_, answer_, not_use_regex_=False):
    for rank_, entry in enumerate(prediction_):
        if not_use_regex_:
            match_fn = exact_match_score
        else:
            match_fn = regex_match_score
        exact_match = metric_max_over_ground_truths(match_fn, normalize(entry['span']), answer_)
        if exact_match:
            return rank_ + 1
    return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_file', type=str)
    parser.add_argument('-p', '--prediction_file', type=str)
    parser.add_argument('-d', '--doc_id_file', type=str)
    parser.add_argument('-nr', '--no_regex', action='store_true', help='default to use regex match')
    parser.add_argument('-tn', '--top_n', type=int, default=150, help='print top n accuracy')

    args = parser.parse_args()
    answer_file = args.answer_file
    prediction_file = args.prediction_file
    prediction_file_prefix = os.path.splitext(prediction_file)[0]
    qa_dict = dict()
    question_count = 1

    with open(args.doc_id_file, encoding=ENCODING) as f:
        id_docs = json.load(f)
    with open(prediction_file_prefix + '.features.log', 'w', encoding=ENCODING) as f:
        for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                              open(prediction_file, encoding=ENCODING)):
            data = json.loads(data_line)
            question = data['question']
            answer = [normalize(a) for a in data['answer']]
            prediction = json.loads(prediction_line)
            answer_predictions = sorted(prediction, key=lambda k: k['doc_score'], reverse=True)
            prediction_rank = get_rank(answer_predictions, answer, args.no_regex)
            qa_predictions = {
                'question': question,
                'rank': prediction_rank if prediction_rank < args.top_n else -1,
                'answer': answer
            }
            # qa_prediction.update(answer_predictions[prediction_rank - 1 if prediction_rank <= args.top_n else 0])

            qa_prediction = []
            qa_str = 'q_{}: {}\n'.format(question_count, question)
            qa_str += 'predicted_rank: {}, answer: {}\n'.format(prediction_rank, '; '.join(answer))
            print(qa_str)
            for d_no, ans_prediction in enumerate(answer_predictions, 1):

                ans_doc = id_docs[ans_prediction['doc_id']]
                nlp_doc = nlp(ans_doc)
                ans_sentences = [s.text for s in nlp_doc.sents]

                ans_sent_idx = -1
                len_sum = 0
                for s_no, s in enumerate(nlp_doc.sents):
                    len_sum += len(s)
                    if len_sum >= ans_prediction['start']:
                        ans_sent_idx = s_no
                        break

                ans_prediction['sentences'] = ans_sentences  # list of sentences
                ans_prediction['ans_sent_idx'] = ans_sent_idx  # answer sentence index

                qa_str += '\tdoc_{:3s}={:12s}={:.4f}, a_score={:.4f}, ans={:20s}, s={}, e={}, sent={}\n'.format(
                    str(d_no), ans_prediction['doc_id'], ans_prediction['doc_score'], ans_prediction['span_score'],
                    ans_prediction['span'], ans_prediction['start'], ans_prediction['end'],
                    ans_prediction['sentences'][ans_prediction['ans_sent_idx']])

                qa_prediction.append(ans_prediction)
            f.write(qa_str + '\n')
            f.flush()
            qa_predictions['predictions'] = qa_prediction
            qa_dict[question_count] = qa_predictions
            question_count += 1
    with open(prediction_file_prefix + '.features.json', 'w', encoding=ENCODING) as f:
        f.write(json.dumps(qa_dict, sort_keys=True, indent=2))
