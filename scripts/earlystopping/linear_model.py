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

    def __init__(self, records_, weight_file_, has_answer=False):
        self.records_ = records_
        self.has_answer = has_answer
        logger.info('Loading model %s' % weight_file_)
        saved_params = torch.load(weight_file_, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        emb_weights = state_dict['embedding.weight']
        self.embedding = nn.Embedding(emb_weights.size(0), emb_weights.size(1), padding_idx=0)
        self.embedding.weight = nn.Parameter(emb_weights)
        self.embedding.weight.requires_grad = False

    def __len__(self):
        return len(self.records_)

    def __getitem__(self, index):
        return self.vectorize(self.records_[index],  self.has_answer)

    def vectorize(self, record_, has_answer=False):
        a_s = record_['a_s']
        a_s_t = torch.FloatTensor([a_s])

        d_s = record_['d_s']
        d_s_t = torch.FloatTensor([d_s])

        a_idx = record_['a_idx']
        a_idx_t = torch.LongTensor(a_idx)
        a_emb = self.embedding(a_idx_t)

        p_ner = record_['p_ner']
        p_ner_t = torch.zeros(len(p_ner), len(Tokenizer.NER))
        for i, w in enumerate(p_ner):
            if w in Tokenizer.NER_DICT:
                p_ner_t[i][Tokenizer.NER_DICT[w]] = 1.0

        p_pos = record_['p_pos']
        p_pos_t = torch.zeros(len(p_pos), len(Tokenizer.POS))
        for i, w in enumerate(p_pos):
            if w in Tokenizer.POS_DICT:
                p_pos_t[i][Tokenizer.POS_DICT[w]] = 1.0

        p_tf = record_['p_tf']
        p_tf_t = torch.FloatTensor(p_tf)

        p_idx = record_['p_idx']
        p_idx_t = torch.LongTensor(p_idx)
        p_emb = self.embedding(p_idx_t)

        p_f = torch.cat([p_emb, p_ner_t, p_pos_t, p_tf_t], dim=2)
        q_ner = record_['q_ner']
        q_ner_t = torch.zeros(len(q_ner), len(Tokenizer.NER))
        for i, w in enumerate(q_ner):
            if w in Tokenizer.NER_DICT:
                q_ner_t[i][Tokenizer.NER_DICT[w]] = 1.0

        q_pos = record_['q_pos']
        q_pos_t = torch.zeros(len(q_pos), len(Tokenizer.POS))
        for i, w in enumerate(q_pos):
            if w in Tokenizer.POS_DICT:
                q_pos_t[i][Tokenizer.POS_DICT[w]] = 1.0

        q_tf = record_['q_tf']
        q_tf_t = torch.FloatTensor(q_tf)

        q_idx = record_['q_idx']
        q_idx_t = torch.LongTensor(q_idx)
        q_emb = self.embedding(q_idx_t)
        q_f = torch.cat([q_emb, q_ner_t, q_pos_t, q_tf_t], dim=2)

        if has_answer:
            label = record_['stop']
            l_t = torch.FloatTensor([label])
            return a_s_t, d_s_t, a_emb, q_f, p_f, l_t
        else:
            return a_s_t, d_s_t, a_emb, q_f, p_f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_file',
                        default='../../data/earlystopping/records-10.txt')
    parser.add_argument('-w', '--weight_file',
                        default='../../data/reader/multitask.mdl')

    args = parser.parse_args()
    NUM_LABELS = 2

    weight_file = args.weight_file

    record_file = args.record_file
    for data_line in open(record_file, encoding=ENCODING):
        record_ = json.loads(data_line)
        print(record_)

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
