#!/usr/bin/env python3
import logging
import math
import os
import pickle as pk

import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from drqa.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)
ENCODING = "utf-8"

H = 32
NUM_CLASS = 2
NLP_NUM = len(Tokenizer.FEAT)
DIM = 7

I_STD = 28.56
I_MEAN = 14.08
Z_STD = 241297
Z_MEAN = 3164

ANS_MEAN = 86486
ANS_STD = 256258


class EarlyStoppingClassifier(nn.Module):

    def __init__(self):
        super(EarlyStoppingClassifier, self).__init__()
        self.fc1 = nn.Linear(DIM, NUM_CLASS)

    def forward(self, input_):
        x = self.fc1(input_)
        return F.log_softmax(x)
        # return F.sigmoid(x)


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
        if next(self.network.parameters()).is_cuda:
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
            if dim:
                return torch.exp(score[:, 1])
            #    return score[:, 1]
            else:
                return torch.exp(score)[1]
            #    return score[0]
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
            record_data = None
        # sp = torch.FloatTensor(record_data['sp'])  # 4x1
        # sa = torch.FloatTensor(record_data['sa'])  # 1x1
        # a_zscore = torch.FloatTensor(list([record_data['a_zscore']]))  # 1
        # max_zscore = torch.FloatTensor(list([record_data['max_zscore']]))  # 1
        # abs_a_zscore = torch.FloatTensor(list([abs(record_data['a_zscore'])]))  # 1

        if record_data['max_zscore'] > 0:
            log_max_zscore = math.log(record_data['max_zscore'])
        else:
            log_max_zscore = 0

        log_max_zscore = torch.FloatTensor([log_max_zscore])

        # az_norm = (record_data['a_zscore'] - Z_MEAN) / Z_STD
        # if record_data['a_zscore'] != 0:
            # a_zscore_norm = torch.FloatTensor(list([az_norm]))  # 1
        # else:
            # a_zscore_norm = a_zscore

        corr_doc_score = torch.FloatTensor(list([record_data['corr_doc_score']]))  # 1

        # Uncomment later
        # np = torch.FloatTensor(record_data['np'])  # 4x58
        # na = torch.FloatTensor(record_data['na'])  # 4x58
        # nq = torch.FloatTensor(record_data['nq'])  # 1x58
        # i_ft = torch.FloatTensor([record_data['i']])  # 1x58

        # repeats = torch.FloatTensor([record_data['repeats']])

        # i_std = (record_data['i'] - I_MEAN) / I_STD
        # i_std = torch.FloatTensor([i_std])

        # prob_avg = torch.FloatTensor([record_data['prob_avg']])  # 1x58

        # ans_avg = torch.FloatTensor([record_data['ans_avg']])

        repeats_2 = 1 if record_data['repeats'] == 2 else 0
        repeats_3 = 1 if record_data['repeats'] == 3 else 0
        repeats_4 = 1 if record_data['repeats'] == 4 else 0
        repeats_5 = 1 if record_data['repeats'] >= 5 else 0
        past20 = 1 if record_data['i'] >= 20 else 0

        repeats_2 = torch.FloatTensor([repeats_2])
        repeats_3 = torch.FloatTensor([repeats_3])
        repeats_4 = torch.FloatTensor([repeats_4])
        repeats_5 = torch.FloatTensor([repeats_5])
        past20 = torch.FloatTensor([past20])

        # FINAL FEATS
        ft = torch.cat([corr_doc_score, log_max_zscore, repeats_2, repeats_3, repeats_4, repeats_5, past20])

        if has_label:
            label = record_data['stop']
            lt = torch.LongTensor([label])
            return ft, lt
        else:
            return ft
