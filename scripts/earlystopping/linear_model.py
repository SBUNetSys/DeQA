#!/usr/bin/env python3
import json
import argparse
import os
import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
import logging
import time
import gc
from drqa.tokenizers.tokenizer import Tokenizer
from drqa.reader import utils
from drqa.pipeline import DEFAULTS

logger = logging.getLogger(__name__)
ENCODING = "utf-8"
NUM_CLASS = 2
MAX_A_LEN = 15
MAX_Q_LEN = 34
MAX_P_LEN = 2683
NLP_NUM = len(Tokenizer.NER) + len(Tokenizer.POS)
DIM = 2 + MAX_A_LEN * NLP_NUM + MAX_Q_LEN * NLP_NUM + MAX_P_LEN * (NLP_NUM + 1)


class EarlyStoppingClassifier(nn.Module):

    def __init__(self):
        super(EarlyStoppingClassifier, self).__init__()
        self.linear = nn.Linear(DIM, NUM_CLASS)

    def forward(self, input_):
        return F.log_softmax(self.linear(input_), dim=0)


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
            target = Variable(ex[1].cuda(async=True))
        else:
            inputs = Variable(ex[0])
            target = Variable(ex[1])

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
            correct += (preds_.numpy() == labels_.numpy()).sum()

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


def batchify(batch_):
    NUM_INPUTS = 5
    NUM_LABEL = 1
    length = len(batch_)
    ans_scores = [one[0] for one in batch_]
    ans_t = torch.zeros(length, 1)
    for i, d in enumerate(ans_scores):
        ans_t[i, :d.size(0)].copy_(d)

    doc_scores = [one[1] for one in batch_]
    doc_t = torch.zeros(length, 1)
    for i, d in enumerate(doc_scores):
        doc_t[i, :d.size(0)].copy_(d)

    answer_feature = [one[2] for one in batch_]
    a_f_t = torch.zeros(length, MAX_A_LEN * NLP_NUM)
    for i, d in enumerate(answer_feature):
        a_f_t[i].copy_(d.view(MAX_A_LEN * NLP_NUM))

    question_feature = [one[3] for one in batch_]
    q_f_t = torch.zeros(length, MAX_Q_LEN * NLP_NUM)
    for i, d in enumerate(question_feature):
        q_f_t[i].copy_(d.view(MAX_Q_LEN * NLP_NUM))

    paragraph_feature = [one[4] for one in batch_]
    p_f_t = torch.zeros(length, MAX_P_LEN * (NLP_NUM + 1))
    for i, d in enumerate(paragraph_feature):
        p_f_t[i].copy_(d.view(MAX_P_LEN * (NLP_NUM + 1)))

    if len(batch_[0]) == NUM_INPUTS:
        return torch.cat([ans_t.view(length, -1), doc_t.view(length, -1), a_f_t.view(length, -1),
                      q_f_t.view(length, -1), p_f_t.view(length, -1)], dim=1)
    elif len(batch_[0]) == (NUM_INPUTS + NUM_LABEL):
        label = [one[5] for one in batch_]
        l_t = torch.LongTensor(len(label), 1).zero_()
        for i, d in enumerate(label):
            l_t[i].copy_(d)
    else:
        raise RuntimeError('Incorrect number of inputs per batch')
    return torch.cat([ans_t.view(length, -1), doc_t.view(length, -1), a_f_t.view(length, -1),
                      q_f_t.view(length, -1), p_f_t.view(length, -1)], dim=1), l_t.view(length)


class RecordDataset(Dataset):

    def __init__(self, records_, has_answer=False):
        self.records_ = records_
        self.has_answer = has_answer
        # logger.info('Loading model %s' % weight_file_)
        # saved_params = torch.load(weight_file_, map_location=lambda storage, loc: storage)
        # state_dict = saved_params['state_dict']
        # emb_weights = state_dict['embedding.weight']
        # self.embedding = nn.Embedding(emb_weights.size(0), emb_weights.size(1), padding_idx=0)
        # self.embedding.weight = nn.Parameter(emb_weights)
        # self.embedding.weight.requires_grad = False

    def __len__(self):
        return len(self.records_)

    def __getitem__(self, index):
        return self.vectorize(self.records_[index], self.has_answer)

    def vectorize(self, record_, has_label=False):
        """
        vectorize a data record
        :param record_: data record dict, example:
        {"a": "2", "a_idx": [499], "a_s": 19766376.0, "d_s": -7.04479736,
        "p_idx": [212316, 155218, ..., 608], "p_ner": ["O", "PERSON", "PERSON", ... , "ORGANIZATION"],
        "p_pos": ["JJ", "NNS", "VBP", ..., "NNP", "-RRB-", "."], "p_tf": [0.005495, 0.005495, ... , 0.010989],
        "q": "How many Grammys has Lady Gaga won?",
        "q_idx": [252, 531, 62637, 613, 57941, 176484, 526, 144],
        "q_ner": ["O", "O", "O", "O", "PERSON", "PERSON", "O", "O"],
        "q_pos": ["WRB", "JJ", "NNPS", "VBZ", "NNP", "NNP", "VBD", "."],
        "q_tf": [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
        "stop": 0}
        :param has_label: whether dataset has label or not
        :return: vectorized records: a_s_t, d_s_t, a_emb, q_f, p_f, l_t(if has label)
        """
        a_s = record_['a_s']
        a_s_t = torch.FloatTensor([a_s])

        d_s = record_['d_s']
        d_s_t = torch.FloatTensor([d_s])

        doc_id = record_['doc_id']
        doc_path = os.path.join(DEFAULTS['features'], '%s.json' % doc_id)
        if os.path.exists(doc_path):
            doc_data = open(doc_path, encoding=ENCODING).read()
            feature = json.loads(doc_data)
            record_['p_tf'] = feature['tf']
            record_['p_ner'] = feature['ner']
            record_['p_pos'] = feature['pos']

            # a_idx = feature['idx'][int(s):int(e) + 1]
            # record['a_idx'] = a_idx
        else:
            print('warning: %s not exist!' % doc_path)

        p_ner = record_['p_ner']
        p_ner_t = torch.zeros(MAX_P_LEN, len(Tokenizer.NER))
        for i, w in enumerate(p_ner):
            if w in Tokenizer.NER_DICT:
                p_ner_t[i][Tokenizer.NER_DICT[w]] = 1.0

        p_pos = record_['p_pos']
        p_pos_t = torch.zeros(MAX_P_LEN, len(Tokenizer.POS))
        for i, w in enumerate(p_pos):
            if w in Tokenizer.POS_DICT:
                p_pos_t[i][Tokenizer.POS_DICT[w]] = 1.0

        s, e = record_['a_loc']
        a_ner = p_ner[s:e+1]
        a_ner_t = torch.zeros(MAX_A_LEN, len(Tokenizer.NER))
        for i, w in enumerate(a_ner):
            if w in Tokenizer.NER_DICT:
                a_ner_t[i][Tokenizer.NER_DICT[w]] = 1.0

        a_pos = p_pos[s:e+1]
        a_pos_t = torch.zeros(MAX_A_LEN, len(Tokenizer.POS))
        for i, w in enumerate(a_pos):
            if w in Tokenizer.POS_DICT:
                a_pos_t[i][Tokenizer.POS_DICT[w]] = 1.0

        a_f = torch.cat([a_ner_t, a_pos_t], dim=1)

        p_tf = record_['p_tf']
        p_tf_t = torch.zeros(MAX_P_LEN)
        p_tf_t[:len(p_tf)].copy_(torch.FloatTensor(p_tf))

        # p_idx = record_['p_idx']
        # p_idx_t = torch.LongTensor(p_idx)
        # p_emb = self.embedding(p_idx_t)

        p_f = torch.cat([p_ner_t, p_pos_t, p_tf_t.view(-1, 1)], dim=1)
        q_ner = record_['q_ner']
        q_ner_t = torch.zeros(MAX_Q_LEN, len(Tokenizer.NER))
        for i, w in enumerate(q_ner):
            if w in Tokenizer.NER_DICT:
                q_ner_t[i][Tokenizer.NER_DICT[w]] = 1.0

        q_pos = record_['q_pos']
        q_pos_t = torch.zeros(MAX_Q_LEN, len(Tokenizer.POS))
        for i, w in enumerate(q_pos):
            if w in Tokenizer.POS_DICT:
                q_pos_t[i][Tokenizer.POS_DICT[w]] = 1.0

        # q_tf = record_['q_tf']
        # q_tf_t = torch.zeros(MAX_Q_LEN, 1)
        # q_tf_t[:len(p_tf), 0].copy_(torch.FloatTensor(q_tf).view(-1, 1))

        # q_idx = record_['q_idx']
        # q_idx_t = torch.LongTensor(q_idx)
        # q_emb = self.embedding(q_idx_t)
        q_f = torch.cat([q_ner_t, q_pos_t], dim=1)

        if has_label:
            label = record_['stop']
            l_t = torch.LongTensor([label])
            return a_s_t, d_s_t, a_f, q_f, p_f, l_t
        else:
            return a_s_t, d_s_t, a_f, q_f, p_f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_file',
                        default='../../data/earlystopping/records-10.txt')
    # parser.add_argument('-w', '--weight_file', default='../../data/reader/multitask.mdl')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Train on CPU, even if GPUs are available.')
    parser.add_argument('--data_workers', type=int, default=os.cpu_count()/2,
                        help='Number of subprocesses for data loading')
    parser.add_argument('--parallel', action='store_true',
                        help='Use DataParallel on all available GPUs')
    parser.add_argument('--random_seed', type=int, default=1013,
                        help=('Random seed for all numpy/torch/cuda '
                              'operations (for reproducibility)'))
    parser.add_argument('--epochs', type=int, default=40,
                        help='Train data iterations')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--optimizer', type=str, default='adamax',
                        help='Optimizer: sgd or adamax')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for SGD only')
    parser.add_argument('--grad_clipping', type=float, default=10,
                        help='Gradient clipping')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay factor')
    parser.add_argument('--split_ratio', type=float, default=0.7,
                        help='ratio of train/dev')
    parser.add_argument('--momentum', type=float, default=0,
                        help='Momentum factor')
    parser.add_argument('--model_file', type=str, default=DEFAULTS['linear_model'],
                        help='Unique model identifier (.mdl, .txt, .checkpoint)')
    parser.add_argument('--checkpoint', type=bool, default=True,
                        help='Save model + optimizer state after each epoch')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # Set random state
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    record_file = args.record_file
    records = []
    logger.info('reading data records: %s' % record_file)
    for data_line in open(record_file, encoding=ENCODING):
        record = json.loads(data_line)
        records.append(record)

    divider = int(args.split_ratio * len(records))
    train_dataset = RecordDataset(records[:divider], has_answer=True)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=batchify,
        pin_memory=args.cuda,
    )
    dev_dataset = RecordDataset(records[divider:], has_answer=False)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=batchify,
        pin_memory=args.cuda,
    )

    model = EarlyStoppingModel(args)
    model.init_optimizer()

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 50)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    best_acc = 0
    for epoch in range(0, args.epochs):
        stats['epoch'] = epoch
        train_loss = utils.AverageMeter()
        epoch_time = utils.Timer()
        # Run one epoch
        for idx, ex in enumerate(train_loader):
            train_loss.update(*model.update(ex))

            if idx % 100 == 0:
                logger.info('epoch: %d, iter = %d/%d | ' %
                            (stats['epoch'], idx, len(train_loader)) +
                            'loss = %.2f, elapsed time = %.2f (s)' %
                            (train_loss.avg, stats['timer'].time()))
                train_loss.reset()
                gc.collect()

        train_acc = model.eval(train_loader)
        dev_acc = model.eval(dev_loader)
        if dev_acc > best_acc:
            best_acc = dev_acc
            model.save(args.model_file)
        logger.info('Epoch %d took %.2f (s), train acc: %.2f, dev acc: %.2f '
                    % (stats['epoch'], epoch_time.time(), train_acc, dev_acc))
        if args.checkpoint:
            model.checkpoint(args.model_file + '.checkpoint', epoch + 1)
