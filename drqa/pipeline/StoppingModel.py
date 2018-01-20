#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.autograd import Variable
import logging
import pickle as pk
from drqa.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)
ENCODING = "utf-8"
H = 32
NUM_CLASS = 2
NLP_NUM = len(Tokenizer.FEAT)
DIM = 1 * 4 + 4 * NLP_NUM * 2 + NLP_NUM


class EarlyStoppingClassifier(nn.Module):

    def __init__(self):
        super(EarlyStoppingClassifier, self).__init__()
        self.fc1 = nn.Linear(DIM, NUM_CLASS)

    def forward(self, input_):
        x = self.fc1(input_)
        return F.log_softmax(x)


class EarlyStoppingModel(object):

    def __init__(self, args_, state_dict_=None):
        self.updates = 0
        self.args = args_
        self.network = EarlyStoppingClassifier()
        if state_dict_:
            self.network.load_state_dict(state_dict_)
        if self.args.cuda and torch.cuda.is_available():
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
        # Update parameters
        self.optimizer.step()
        self.updates += 1
        return loss.data[0], ex[0].size(0)

    def predict(self, inputs, prob=False):
        self.network.eval()
        if self.args.cuda and torch.cuda.is_available():
            inputs_var = Variable(inputs.cuda(async=True))
        else:
            inputs_var = Variable(inputs)
        # Run forward
        score_ = self.network(inputs_var)
        score = score_.data.cpu()
        dim = 0 if len(score.size()) == 1 else 1
        _, predicted_score = torch.max(score, dim)
        if prob:
            # return stop probability
            return torch.exp(score[:, 1])
        else:
            return predicted_score

    def eval(self, data_loader_):
        all_tp = 0
        all_fp = 0
        all_fn = 0
        all_tn = 0
        for batch_ in data_loader_:
            preds_ = self.predict(batch_[0])
            labels_ = torch.squeeze(batch_[1])
            tp = (preds_ * labels_).sum()
            tn = ((1 - preds_) * (1 - labels_)).sum()
            fp = (preds_ * (1 - labels_)).sum()
            fn = ((1 - preds_) * labels_).sum()
            all_tn += tn
            all_fp += fp
            all_fn += fn
            all_tp += tp

        precision_ = all_tp / (all_tp + all_fp + 1e-6) * 100
        recall_ = all_tp / (all_tp + all_fn + 1e-6) * 100
        accuracy_ = (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn + 1e-6) * 100
        f1_ = 2 * precision_ * recall_ / (precision_ + recall_ + 1e-6)
        return accuracy_, precision_, recall_, f1_

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
        sp = torch.FloatTensor(record_data['sp'])  # 4x1
        # sa = torch.FloatTensor(record_data['sa'])  # 4x1

        np = torch.FloatTensor(record_data['np'])  # 4x58
        na = torch.FloatTensor(record_data['na'])  # 4x58
        nq = torch.FloatTensor(record_data['nq'])  # 1x58

        # hq = torch.FloatTensor(record_data['hq'])
        # hp = torch.FloatTensor(record_data['hp'])
        # ha = torch.FloatTensor(record_data['ha'])  # 4x768

        # ft = torch.cat([sp, sa, nq, np, na, hq, hp, ha])
        ft = torch.cat([sp, nq, np, na])

        if has_label:
            label = record_data['stop']
            lt = torch.LongTensor([label])
            return ft, lt
        else:
            return ft
