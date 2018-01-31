#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from .. import DATA_DIR

DEFAULTS = {
    'sql_path': os.path.join(DATA_DIR, 'wikipedia/wiki_idx.db'),
    'db_path': os.path.join(DATA_DIR, 'wikipedia/docs.db'),
    'tfidf_path': os.path.join(DATA_DIR, 'wikipedia/tfidf-'),
    'galago_path': os.path.join(DATA_DIR, 'galago/bin/galago'),
    'galago_index': os.path.join(DATA_DIR, 'wikipedia/wiki-idx-20171103/')
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'galago':
        return GalagoRanker
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sql':
        return SqliteRanker
    if name == 'lucene':
        return LuceneRanker
    raise RuntimeError('Invalid retriever class: %s' % name)


from .tfidf_doc_ranker import TfidfDocRanker
from .galago_ranker import GalagoRanker
from .sqlite_ranker import SqliteRanker
from .lucene_ranker import LuceneRanker
