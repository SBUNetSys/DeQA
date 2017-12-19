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
from drqa.tokenizers.tokenizer import Tokenizer
from drqa.reader import utils

logger = logging.getLogger(__name__)
ENCODING = "utf-8"
NUM_CLASS = 2
MAX_A_LEN = 15
MAX_Q_LEN = 20
MAX_P_LEN = 800
NLP_NUM = len(Tokenizer.NER) + len(Tokenizer.POS)
DIM = 2 + MAX_A_LEN * NLP_NUM + MAX_Q_LEN * NLP_NUM + MAX_P_LEN * (NLP_NUM + 1)


class EarlyStoppingClassifier(nn.Module):

    def __init__(self):
        super(EarlyStoppingClassifier, self).__init__()
        self.linear = nn.Linear(DIM, NUM_CLASS)

    def forward(self, input_):
        return F.log_softmax(self.linear(input_))


class EarlyStoppingModel(object):

    def __init__(self, args_):
        self.updates = 0
        self.args = args_
        self.network = EarlyStoppingClassifier()

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

    def predict(self, ex):
        self.network.eval()
        # Transfer to GPU
        if self.args.cuda:
            inputs = [e if e is None else
                      Variable(e.cuda(async=True), volatile=True)
                      for e in ex[:5]]
        else:
            inputs = [e if e is None else Variable(e, volatile=True)
                      for e in ex[:5]]
        # Run forward
        score_ = self.network(*inputs)
        score = score_.data.cpu()
        return 1 if score > 0.5 else 0

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
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='Train on CPU, even if GPUs are available.')
    parser.add_argument('--data_workers', type=int, default=5,
                        help='Number of subprocesses for data loading')
    parser.add_argument('--parallel', type=bool, default=False,
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
    parser.add_argument('--momentum', type=float, default=0,
                        help='Momentum factor')
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
    for data_line in open(record_file, encoding=ENCODING):
        record = json.loads(data_line)
        records.append(record)

    # max_a_len = max([len(r['a_idx']) for r in records])
    # max_q_len = max([len(r['q_idx']) for r in records])
    # max_p_len = max([len(r['p_idx']) for r in records])

    train_dataset = RecordDataset(records[:-1000], has_answer=True)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=batchify,
        pin_memory=args.cuda,
    )

    dev_dataset = RecordDataset(records[-1000:], has_answer=False)
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
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    for epoch in range(0, args.epochs):
        stats['epoch'] = epoch
        train_loss = utils.AverageMeter()
        epoch_time = utils.Timer()
        # Run one epoch
        for idx, ex in enumerate(train_loader):
            train_loss.update(*model.update(ex))

            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (stats['epoch'], idx, len(train_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, stats['timer'].time()))
            train_loss.reset()

        logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                    (stats['epoch'], epoch_time.time()))

        # Validate unofficial (train)
        # validate_unofficial(args, train_loader, model, stats, mode='train')

    # Checkpoint
    # if args.checkpoint:
    #     model.checkpoint(args.model_file + '.checkpoint',
    #                      global_stats['epoch'] + 1)

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
