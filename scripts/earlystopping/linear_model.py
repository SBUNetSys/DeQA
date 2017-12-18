#!/usr/bin/env python3
import json
import argparse
import os
import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
import logging
import time
from drqa.tokenizers.tokenizer import Tokenizer
logger = logging.getLogger(__name__)
ENCODING = "utf-8"


class EarlyStoppingModel(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        super(EarlyStoppingModel, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec))


def batchify(batch_):

    return torch.LongTensor(batch_)


class RecordDataset(Dataset):

    def __init__(self, records_, has_answer=False):
        self.records_ = records_
        self.has_answer = has_answer
        # logger.info('Loading model %s' % weights_file)
        # saved_params = torch.load(weights_file, map_location=lambda storage, loc: storage)
        # self.word_dict = saved_params['word_dict']
        # self.state_dict = saved_params['state_dict']

    def __len__(self):
        return len(self.records_)

    def __getitem__(self, index):
        return self.vectorize(self.records_[index],  self.has_answer)

    def vectorize(self, record_, has_answer=False):
        answer = record_['a']
        answer_score = record_['a_s']
        doc_score = record_['d_s']
        question = record_['q']
        ner = record_['ner']
        pos = record_['pos']
        tf = record_['tf']
        if has_answer:
            label = record_['stop']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_file',
                        default='../../data/earlystopping/records-10.txt')

    args = parser.parse_args()
    NUM_LABELS = 2

    record_file = args.record_file
    for data_line in open(record_file, encoding=ENCODING):
        data = json.loads(data_line)

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
