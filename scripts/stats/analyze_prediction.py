#!/usr/bin/env python3
import argparse
import ujson as json
from drqa.retriever.utils import normalize
from drqa.reader.utils import exact_match_score, regex_match_score, metric_max_over_ground_truths

ENCODING = "utf-8"


def get_rank(prediction_, answer_, use_regex_=False):
    for rank_, entry in enumerate(prediction_):
        if use_regex_:
            match_fn = regex_match_score
        else:
            match_fn = exact_match_score
        exact_match = metric_max_over_ground_truths(match_fn, normalize(entry['span']), answer_)
        if exact_match:
            return rank_ + 1
    return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--answer_file', type=str, default='data/datasets/SQuAD-v1.1-dev.txt')
    parser.add_argument('-p', '--prediction_file', default='data/earlystopping/SQuAD-v1.1-dev-multitask-pipeline.preds')
    parser.add_argument('-ans', '--answer_rank', action='store_true', help='default to use doc score rank')
    parser.add_argument('-r', '--regex', action='store_true', help='default to use exact match')
    parser.add_argument('-d', '--draw', action='store_true', help='default not output draw data')
    parser.add_argument('-s', '--stop_location', type=int, default=151, help='manual stop location')
    parser.add_argument('-t', '--top_n', type=int, default=150, help='print top n accuracy')

    args = parser.parse_args()
    answer_file = args.answer_file
    prediction_file = args.prediction_file

    qa_dict = dict()
    question_count = 1
    with open(prediction_file + '.readable-ans.log', 'w', encoding=ENCODING) as f:
        with open(prediction_file + '.perfect.txt', 'w', encoding=ENCODING) as pf:
            for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
                                                  open(prediction_file, encoding=ENCODING)):
                data = json.loads(data_line)
                question = data['question']
                answer = [normalize(a) for a in data['answer']]
                prediction = json.loads(prediction_line)
                prediction = sorted(prediction, key=lambda k: k['doc_score'], reverse=True)
                doc_predictions = prediction[:args.stop_location]
                if args.answer_rank:
                    answer_predictions = sorted(doc_predictions, key=lambda k: k['span_score'], reverse=True)
                    prediction_rank = get_rank(answer_predictions, answer, args.regex)
                else:
                    # doc rank
                    answer_predictions = doc_predictions
                    prediction_rank = get_rank(answer_predictions, answer, args.regex)
                qa_prediction = {
                    'question': question,
                    'rank': prediction_rank if prediction_rank < args.top_n else -1,
                    'answer': answer
                }
                qa_prediction.update(answer_predictions[prediction_rank - 1 if prediction_rank <= args.top_n else 0])
                qa_dict[question_count] = qa_prediction
                qa_str = 'q_{}: {}\n'.format(question_count, question)
                qa_str += 'predicted_rank: {}, answer: {}\n'.format(prediction_rank, '; '.join(answer))
                for d_no, ans_prediction in enumerate(answer_predictions, 1):
                    qa_str += '\tdoc_{:3s}: {:12s}, d_score: {:.4f}, a_score: {:.4f}, ans: {:20s}, s: {}, e: {}\n'.format(
                        str(d_no), ans_prediction['doc_id'], ans_prediction['doc_score'], ans_prediction['span_score'],
                        ans_prediction['span'], ans_prediction['start'], ans_prediction['end'])

                f.write(qa_str + '\n')
                pf.write(json.dumps(answer_predictions[:prediction_rank]) + '\n')
                question_count += 1

    file_suffix = '.qa-a.json' if args.answer_rank else '.qa-d.json'
    with open(prediction_file + file_suffix, 'w', encoding=ENCODING) as f:
        f.write(json.dumps(qa_dict, sort_keys=True, indent=2))
