#!/usr/bin/env python3
import argparse
import os
import torch.autograd
import logging
import random
import glob
from utils import Timer, AverageMeter, exact_match_score, regex_match_score
from StoppingModel import EarlyStoppingModel, RecordDataset
from multiprocessing import Pool as ProcessPool
from eval_model import batch_predict
import sys

logger = logging.getLogger(__name__)
ENCODING = "utf-8"


if __name__ == '__main__':
    # sample run: python linear_model.py -r records
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_dir', default=None)
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
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for SGD only')
    parser.add_argument('--momentum', type=float, default=0,
                        help='Momentum factor for SGD only')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay factor')
    parser.add_argument('-m', '--model_file', type=str, default='linear_model.mdl',
                        help='Unique model identifier (.mdl, .txt, .checkpoint)')
    parser.add_argument('--checkpoint', type=bool, default=False,
                        help='Save model + optimizer state after each epoch')

    parser.add_argument('-p', '--prediction_file', default='CuratedTrec-test-lstm.preds.txt',
                        help='prediction file, e.g. CuratedTrec-test-lstm.preds.txt')
    parser.add_argument('-a', '--answer_file', default='CuratedTrec-test.txt', help='data set with labels, e.g. CuratedTrec-test.txt')

#    parser.add_argument('-p', '--prediction_file',
#                        help='prediction file, e.g. CuratedTrec-test-lstm.preds.txt')
#    parser.add_argument('-a', '--answer_file', help='data set with labels, e.g. CuratedTrec-test.txt')

    parser.add_argument('-f', '--feature_dir', default=None,
                        help='dir that contains json features files, unzip squad.tgz or trec.tgz to get that dir')
    parser.add_argument('-rg', '--regex', action='store_true', help='default to use exact match')

    args = parser.parse_args()
    match_func = regex_match_score if args.regex else exact_match_score

    answer_file = args.answer_file
    prediction_file = args.prediction_file

    feature_dir = args.feature_dir
    if not os.path.exists(feature_dir):
        print('feature_dir does not exist!')
        exit(-1)

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

    train_records = glob.glob("%s/*.pkl" % args.record_dir)
    logger.info('found %d records for training' % len(train_records))
    random.shuffle(train_records)
    train_dataset = RecordDataset(train_records, has_answer=True)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        pin_memory=args.cuda,
    )

    model = EarlyStoppingModel(args)
    model.init_optimizer()

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 50)
    logger.info('Starting training...')
    stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0}
    best_acc = 0
    best_epoch = 0
    for epoch in range(0, args.epochs):
        stats['epoch'] = epoch
        train_loss = AverageMeter()
        epoch_time = Timer()
        # Run one epoch
        for idx, ex in enumerate(train_loader):
            train_loss.update(*model.update(ex))
        print("DONE WITH EPOCH")

        train_metric = model.eval(train_loader)
        model.save(args.model_file + '.train')

        eval_model = EarlyStoppingModel.load(args.model_file + '.train')
        eval_model.network.cpu()
        total_count = 0
        correct_count = 0
        result_handles = []
        #async_pool = ProcessPool()

#        for data_line, prediction_line in zip(open(answer_file, encoding=ENCODING),
#                                              open(prediction_file, encoding=ENCODING)):
#            param = (data_line, prediction_line, eval_model, feature_dir, match_func)
            #handle = async_pool.apply_async(batch_predict, param)
#            handle = batch_predict(*param)
#            result_handles.append(handle)


#        for result in result_handles:
           # correct, total = result.get()
#            correct, total = result
#            correct_count += correct
#            total_count += total
#            if total_count % 100 == 0:
             #   print('processed %d/%d, %2.4f' % (correct_count, total_count, correct_count / total_count))
#                print('processed %d/%d, %2.4f' % (correct_count, total_count, total_count))
#            sys.stdout.flush()

#        print("Done with Epoch {}".format(epoch))

#        dev_acc = correct_count / total_count
#        if dev_acc > best_acc:
#            best_acc = dev_acc
#            best_epoch = epoch
#            if args.checkpoint:
#                model.checkpoint(args.model_file + '.checkpoint.%.2f_%d' % (best_acc, best_epoch), epoch + 1)
#            model.save(args.model_file)
#        else:
#            os.remove(args.model_file + '.train')
#        logger.info('Epoch %-2d loss:%.4f, train_acc:%.2f, dev_acc:%.2f(best:%.2f at %d), took:%.2f(s), elapsed:%.2f(s)'
#                    % (stats['epoch'], train_loss.avg, train_metric[0], dev_acc, best_acc, best_epoch,
#                       epoch_time.time(), stats['timer'].time()))
        train_loss.reset()



    model_name, ext = os.path.splitext(args.model_file)
    final_model_file = '%s_%s%s' % (model_name, best_acc, ext)
#    os.rename(args.model_file, final_model_file)
    logger.info('best_acc: %.2f' % best_acc)
