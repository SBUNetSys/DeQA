#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from .. import DATA_DIR

DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'wikipedia/docs.db'),
    'tfidf_path': os.path.join(DATA_DIR, 'wikipedia/tfidf-'),
    'galago_path': os.path.join(DATA_DIR, 'galago/bin/galago'),
    'galago_index': os.path.join(DATA_DIR, 'wikipedia/galago-idx')
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'galago':
        return GalagoRanker
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sqlite':
        return DocDB
    raise RuntimeError('Invalid retriever class: %s' % name)


from .doc_db import DocDB
from .galago_db import GalagoDB
from .tfidf_doc_ranker import TfidfDocRanker
from .galago_ranker import GalagoRanker
