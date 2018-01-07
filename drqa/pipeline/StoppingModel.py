#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.autograd import Variable
import logging
import random
import pickle as pk
import glob
import gc
from drqa.tokenizers.tokenizer import Tokenizer
from drqa.reader import utils
from drqa.pipeline import DEFAULTS

logger = logging.getLogger(__name__)
ENCODING = "utf-8"
H = 128
NUM_CLASS = 2
NLP_NUM = len(Tokenizer.FEAT)
DIM = 2 * 4 + 4 * (NLP_NUM * 2 + 768 * 2) + NLP_NUM + 768


class EarlyStoppingClassifier(nn.Module):

    def __init__(self):
        super(EarlyStoppingClassifier, self).__init__()
        self.fc1 = nn.Linear(DIM, NUM_CLASS)

    def forward(self, input_):
        x = self.fc1(input_).clamp(min=0)
        return F.log_softmax(x)


class EarlyStoppingModel(object):

    def __init__(self, args_, state_dict_=None):
        self.updates = 0
        self.args = args_
        self.network = EarlyStoppingClassifier()
        if state_dict_:
            self.network.load_state_dict(state_dict_)
        if self.args.cuda:
            self.network.cuda()
        if self.args.parallel:
            self.network = torch.nn.DataParallel(self.network)

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.args.cuda:
            inputs = Variable(ex[0].cuda(async=True))
            target = Variable(ex[1].squeeze().cuda(async=True))
        else:
            inputs = Variable(ex[0])
            target = Variable(ex[1].squeeze())

        # Run forward
        score_ = self.network(inputs)

        # Compute loss and accuracies

        loss = F.nll_loss(score_, target)

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # # Reset any partially fixed parameters (e.g. rare words)
        # self.reset_parameters()

        return loss.data[0], ex[0].size(0)

    def predict(self, inputs):
        self.network.eval()
        if self.args.cuda:
            inputs_var = Variable(inputs.cuda(async=True))
        else:
            inputs_var = Variable(inputs)
        # Run forward
        score_ = self.network(inputs_var)
        score = score_.data.cpu()
        dim = 0 if len(score.size()) == 1 else 1
        _, pred = torch.max(score, dim)
        return pred

    def eval(self, data_loader_):
        total = 0
        correct = 0
        for batch_ in data_loader_:
            batch_size = batch_[0].size(0)
            total += batch_size
            preds_ = self.predict(batch_[0])
            labels_ = batch_[1]
            correct += (preds_.numpy().flatten() == labels_.numpy().flatten()).sum()

            # sample max 10k
            if total >= 1e4:
                break
        return correct / total * 100

    def init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    def checkpoint(self, file_name, epoch_):
        params = {
            'state_dict': self.network.state_dict(),
            'args': self.args,
            'epoch': epoch_,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, file_name)
        except Exception as e:
            logger.warning('WARN: %s Saving failed... continuing anyway.' % e)

    def save(self, file_name):
        params = {'state_dict': self.network.state_dict(), 'args': self.args}
        try:
            torch.save(params, file_name)
        except Exception as e:
            logger.warning('WARN: %s Saving failed... continuing anyway.' % e)

    @staticmethod
    def load(filename):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(filename, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        args_ = saved_params['args']
        return EarlyStoppingModel(args_, state_dict)


class RecordDataset(Dataset):

    def __init__(self, records_, has_answer=False):
        self.records_ = records_
        self.has_answer = has_answer

    def __len__(self):
        return len(self.records_)

    def __getitem__(self, index):
        return self.vectorize(self.records_[index], self.has_answer)

    def vectorize(self, record_, has_label=False):
        """
        vectorize a data record
        :param record_: data record file path
        :param has_label: whether dataset has label or not
        :return: vectorized records: a_s_t, d_s_t, a_emb, q_f, p_f, l_t(if has label)
        """

        if os.path.exists(record_):
            record_data = pk.load(open(record_, "rb"))
        else:
            print('warning: %s not exist!' % record_)
        sp = torch.FloatTensor(record_data['sp'])
        sa = torch.FloatTensor(record_data['sa'])

        np = torch.FloatTensor(record_data['np'])
        na = torch.FloatTensor(record_data['na'])
        nq = torch.FloatTensor(record_data['nq'])

        hq = torch.FloatTensor(record_data['hq'])
        hp = torch.FloatTensor(record_data['hp'])
        ha = torch.FloatTensor(record_data['ha'])

        ft = torch.cat([sp, sa, nq, np, na, hq, hp, ha])

        if has_label:
            label = record_data['stop']
            lt = torch.LongTensor([label])
            return ft, lt
        else:
            return ft
