#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run predictions using the full DrQA retriever-reader pipeline."""

import argparse
import json
import logging
import os
import sys
import time

import torch

from drqa import pipeline, retriever

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s.%(msecs)03d: [ %(message)s ]', '%m/%d/%Y_%H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('--out-dir', type=str, default=None,
                    help="Directory to write prediction file")
parser.add_argument('--out-suffix', type=str, default=None,
                    help=("Directory to write prediction file to "
                          "(<dataset>-<model>.predictions.txt)"))
parser.add_argument('--reader-model', type=str, default=None,
                    help="Path to trained Document Reader model")
parser.add_argument('--retriever-model', type=str, default=None,
                    help="Path to Document Retriever model (tfidf)")
parser.add_argument('--db_path', type=str, default=None,
                    help='Path to Document DB or index')
parser.add_argument('--n_docs', type=int, default=150,
                    help="Number of docs to retrieve per query")
parser.add_argument('--top_n', type=int, default=150,
                    help="Number of predictions to make per query")
parser.add_argument('--tokenizer', type=str, default='corenlp',
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--no-cuda', action='store_true', help="Use CPU only")
parser.add_argument('--gpu', type=int, default=0,
                    help="Specify GPU device id to use")
parser.add_argument('--parallel', action='store_true',
                    help='Use data parallel (split across gpus)')
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Document paragraph batching size')
parser.add_argument('--predict-batch-size', type=int, default=1,
                    help='Question batching size')
parser.add_argument('--ranker', type=str, default='lucene')
parser.add_argument('--et_threshold', type=float, default=None,
                    help='early stopping threshold')
parser.add_argument("-v", "--verbose", help="log more debug info", action="store_true")

args = parser.parse_args()
if args.verbose:
    logger.setLevel(logging.DEBUG)

t0 = time.time()
# log_filename = ('_'.join(sys.argv) + time.strftime("%Y%m%d-%H%M%S")).replace('/', '_')
# logfile = logging.FileHandler('/tmp/%s.log' % log_filename, 'w')
# logfile.setFormatter(fmt)
# logger.addHandler(logfile)
logger.info('COMMAND: python %s' % ' '.join(sys.argv))

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

if args.ranker.lower().startswith('s'):
    ranker = retriever.get_class('sql')(db_path=args.db_path)
elif args.ranker.lower().startswith('l'):
    ranker = retriever.get_class('lucene')(index_path=args.db_path)
else:
    ranker = retriever.get_class('tfidf')(tfidf_path=args.retriever_model, db_path=args.db_path)

logger.info('Initializing pipeline...')
DrQA = pipeline.DrQA(
    reader_model=args.reader_model,
    tokenizer=args.tokenizer,
    batch_size=args.batch_size,
    cuda=args.cuda,
    data_parallel=args.parallel,
    ranker=ranker,
    num_workers=args.num_workers,
    et_threshold=args.et_threshold
)

# ------------------------------------------------------------------------------
# Read in dataset and make predictions
# ------------------------------------------------------------------------------


logger.info('Loading queries from %s' % args.dataset)
queries = []
for line in open(args.dataset):
    data = json.loads(line)
    queries.append(data['question'])

model_name = os.path.splitext(os.path.basename(args.reader_model or 'default'))[0]
data_name = os.path.splitext(os.path.basename(args.dataset))[0]
out_dir = args.out_dir or os.path.dirname(args.dataset)
os.makedirs(out_dir, exist_ok=True)
outfile = os.path.join(out_dir, args.out_suffix or '{}-{}.predictions.txt'.format(data_name, model_name))

logger.info('Writing results to %s' % outfile)
with open(outfile, 'w') as f:
    batches = [queries[i: i + args.predict_batch_size]
               for i in range(0, len(queries), args.predict_batch_size)]
    for i, batch in enumerate(batches):
        batch_info = '-' * 5 + ' Batch %d/%d ' % (i + 1, len(batches)) + '-' * 5 + ' '
        start_query = queries[i]
        logger.info(batch_info + start_query)
        predictions = DrQA.process(batch, n_docs=args.n_docs, top_n=args.top_n)
        for p in predictions:
            p = sorted(p, key=lambda k: k['doc_score'], reverse=True)
            f.write(json.dumps(p) + '\n')

logger.info('Total time: %.4f' % (time.time() - t0))
