#!/usr/bin/env python3
import json
import argparse
import os
from collections import Counter, OrderedDict
from drqa.retriever.utils import normalize
from drqa.pipeline import DEFAULTS
from drqa import DATA_DIR
import pickle
ENCODING = "utf-8"

from itertools import count

import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable

POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE + 1)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prediction_file',
                        default='../../data/earlystopping/SQuAD-v1.1-dev-1000-multitask-pipeline.preds')
    parser.add_argument('-a', '--answer_file', default='../../data/datasets/SQuAD-v1.1-dev-1000.txt')

    args = parser.parse_args()
    questions = []
    answers = []
    for line in open(args.answer_file, encoding=ENCODING):
        data = json.loads(line)
        question = data['question']
        questions.append(question)
        answer = [normalize(a) for a in data['answer']]
        answers.append(answer)

    prediction_file = args.prediction_file
    predictions = []
    for line in open(prediction_file, encoding=ENCODING):
        prediction_data = json.loads(line)
        predictions.append(prediction_data)
        # ids = [d['doc_id'] for d in prediction_data]
        # counter = Counter(ids)
        # print(len(counter.keys()))
    records = []
    doc_missing_count = 0
    for question, answer, prediction in zip(questions, answers, predictions):
        for entry in prediction:
            doc_id = entry['doc_id']
            record = OrderedDict()
            record['q'] = question
            record['d_id'] = doc_id
            record['a'] = entry['span']
            record['a_s'] = entry['span_score']
            record['d_s'] = entry['doc_score']

            doc_path = os.path.join(DEFAULTS['features'], '%s.json' % doc_id)
            if os.path.exists(doc_path):
                doc_data = open(doc_path, encoding=ENCODING).read()
                feature = json.loads(doc_data)

                record['d_l'] = feature['l_p']
                record['tf'] = feature['tf']
                record['ner'] = feature['ner']
                record['pos'] = feature['pos']
            else:
                print('%s not exist!' % doc_path)
                doc_missing_count += 1
            records.append(record)

    print('%d docs missing' % doc_missing_count)
    # record_path = os.path.join(DATA_DIR, 'records.json')
    # with open(record_path, 'w') as f:
    #     f.write(json.dumps(records, sort_keys=True))
    #
    record_path = os.path.join(DATA_DIR, 'records.pkl')
    with open(record_path, 'wb') as f:
        pickle.dump(records, f)
    # # Define model
    # fc = torch.nn.Linear(W_target.size(0), 1)
    #
    # for batch_idx in count(1):
    #     # Get data
    #     batch_x, batch_y = get_batch()
    #
    #     # Reset gradients
    #     fc.zero_grad()
    #
    #     # Forward pass
    #     output = F.smooth_l1_loss(fc(batch_x), batch_y)
    #     loss = output.data[0]
    #
    #     # Backward pass
    #     output.backward()
    #
    #     # Apply gradients
    #     for param in fc.parameters():
    #         param.data.add_(-0.1 * param.grad.data)
    #
    #     # Stop criterion
    #     if loss < 1e-3:
    #         break
    #
    #     print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
    # print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
    # print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
