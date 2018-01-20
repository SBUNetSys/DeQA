#!/usr/bin/env python3
import argparse
import os
import torch.autograd
import logging
import random
import glob
import gc
from drqa.reader import utils
from drqa.pipeline import DEFAULTS
from drqa.pipeline.StoppingModel import EarlyStoppingModel, RecordDataset

logger = logging.getLogger(__name__)
ENCODING = "utf-8"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_dir', default=DEFAULTS['records'])
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Train on CPU, even if GPUs are available.')
    parser.add_argument('--data_workers', type=int, default=int(os.cpu_count() / 2),
                        help='Number of subprocesses for data loading')
    parser.add_argument('--parallel', action='store_true',
                        help='Use DataParallel on all available GPUs')
    parser.add_argument('--random_seed', type=int, default=1013,
                        help=('Random seed for all numpy/torch/cuda '
                              'operations (for reproducibility)'))
    parser.add_argument('--epochs', type=int, default=200,
                        help='Train data iterations')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--optimizer', type=str, default='adamax',
                        help='Optimizer: sgd or adamax(gives better training performance')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for SGD only')
    parser.add_argument('--momentum', type=float, default=0,
                        help='Momentum factor for SGD only')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay factor')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='ratio of train/dev')
    parser.add_argument('-m', '--model_file', type=str, default=DEFAULTS['linear_model'],
                        help='Unique model identifier (.mdl, .txt, .checkpoint)')
    parser.add_argument('--checkpoint', type=bool, default=False,
                        help='Save model + optimizer state after each epoch')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # Set random state
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s.%(msecs)03d: [ %(message)s ]', '%m/%d/%Y_%H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    records = glob.glob("%s/*.pkl" % args.record_dir)
    logger.info('found %d records' % len(records))
    divider = int(args.split_ratio * len(records))
    random.shuffle(records)
    train_records = records[:divider]
    train_dataset = RecordDataset(train_records, has_answer=True)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        pin_memory=args.cuda,
    )
    if args.eval:
        dev_dataset = RecordDataset(records, has_answer=True)
    else:
        dev_dataset = RecordDataset(records[divider:], has_answer=True)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        pin_memory=args.cuda,
    )

    if args.eval:
        model = EarlyStoppingModel.load(args.model_file)
        logger.info('model loaded, begin evaluating...')
        dev_acc, dev_precision, dev_recall, dev_f1 = model.eval(dev_loader)
        logger.info('eval acc: %.2f, precision: %.2f, recall: %.2f, f1: %.2f'
                    % (dev_acc, dev_precision, dev_recall, dev_f1))
        # csv_path = os.path.join(os.path.expanduser('~'), 'data.csv')
        # with open(csv_path, 'w') as f:
        #     for j, batch_ in enumerate(dev_loader):
        #         batch_x = batch_[0]
        #         batch_y = batch_[1]
        #         for i, b in enumerate(zip(batch_x, batch_y)):
        #             x, y = b
        #             no = i + j * len(batch_x)
        #             line = '%d, %d, %s' % (no, y.numpy(), ','.join(['%.5f' % num for num in x.numpy()]))
        #             f.write(line + '\n')
        #             if no % 100 == 0:
        #                 logger.info('saved no: %d' % no)
        #             if no == 1000:
        #                 exit(0)
        exit(0)
    else:
        model = EarlyStoppingModel(args)
        model.init_optimizer()

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 50)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    best_f1 = 0
    best_epoch = 0
    for epoch in range(0, args.epochs):
        stats['epoch'] = epoch
        train_loss = utils.AverageMeter()
        epoch_time = utils.Timer()
        # Run one epoch
        for idx, ex in enumerate(train_loader):
            train_loss.update(*model.update(ex))

        train_metric = model.eval(train_loader)
        dev_acc, dev_precision, dev_recall, dev_f1 = model.eval(dev_loader)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            model.save(args.model_file)
            if args.checkpoint:
                model.checkpoint(args.model_file + '.checkpoint.%.2f_%d' % (best_f1, best_epoch), epoch + 1)
        logger.info('Epoch %-2d loss:%.4f, train_acc:%.2f, dev_acc:%.2f, '
                    'precision:%.2f, recall:%.2f, f1:%.2f(best:%.2f at %d), took:%.2f(s), elapsed:%.2f(s)'
                    % (stats['epoch'], train_loss.avg, train_metric[0], dev_acc
                       , dev_precision, dev_recall, dev_f1, best_f1, best_epoch,
                       epoch_time.time(), stats['timer'].time()))
        train_loss.reset()

    model_name, ext = os.path.splitext(args.model_file)
    final_model_file = '%s_%s%s' % (model_name, best_f1, ext)
    os.rename(args.model_file, final_model_file)
    logger.info('best_f1: %.2f' % best_f1)
