#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Documents, in a sqlite database."""

import sqlite3
import logging

from multiprocessing.pool import ThreadPool
from functools import partial
from . import DEFAULTS
from .rake import Rake

logger = logging.getLogger(__name__)


class SqliteRanker(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None, ext_path='data/okapi_bm25.sqlext'):
        self.path = db_path or DEFAULTS['sql_path']
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.enable_load_extension(True)
        self.connection.load_extension(ext_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def closest_docs(self, question, k=5):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        keyword_items = Rake().run(question)
        word_queries = []
        for keyword_item in keyword_items:
            keyword, _ = keyword_item
            # keyword_query = '#od:1( %s )' % keyword if ' ' in keyword else keyword
            keyword_query = keyword.replace('-', ' ').replace('^', '')
            word_queries.append('"%s"' % keyword_query)
        query = ' OR '.join(word_queries[:2])

        sql = '''
              select id, okapi_bm25(matchinfo(wiki, 'pcnalx'), 2) as rank, text
              from wiki where wiki match :query
              order by rank desc limit :number
            '''
        search_results = self.connection.execute(sql, {'query': query, 'number': k})
        doc_scores = []
        doc_ids = []
        doc_texts = []
        for row in search_results:
            doc_id, doc_score, doc_text = row
            doc_ids.append(doc_id)
            doc_scores.append(doc_score)
            doc_texts.append(doc_text)
        logger.info('question:%s, query:%s, doc_ids:%s' % (question, query, ';'.join(doc_ids)))
        return doc_ids, doc_scores, doc_texts

    def batch_closest_docs(self, queries, k=5, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results
