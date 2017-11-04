#!/usr/bin/env python3

import logging
import subprocess

from multiprocessing.pool import ThreadPool
from functools import partial
from . import DEFAULTS

logger = logging.getLogger(__name__)


class GalagoRanker(object):
    """Use Galago search engine to retrieve wikipedia articles
    """

    def __init__(self, galago_path=None, index_path=None):
        self.galago_path = galago_path or DEFAULTS['galago_path']
        self.index_path = index_path or DEFAULTS['galago_index']

    def closest_docs(self, query, k=5):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        args = ['--requested=%s' % k, '--casefold=true', '--query=', '#combine(%s)' % query]
        search_results = self._run_galago('batch-search', args)
        doc_scores = []
        doc_ids = []

        ''' example results
        unk-0 Q0 AP890187-3263 1 -4.58285636 galago
        unk-0 Q0 AP890004-5802 2 -4.73404920 galago
        unk-0 Q0 AP890058-1371 3 -4.87260425 galago
        unk-0 Q0 AP892079-4109 4 -4.94346010 galago
        unk-0 Q0 AP891795-0650 5 -4.95236039 galago
        '''
        for result in search_results.split('\n'):
            result_elements = result.split(' ')
            if len(result_elements) < 5:
                print('query failed', query)
                continue
            doc_ids.append(result_elements[2])
            doc_scores.append(result_elements[4])
        print('query:', query, 'docID:', doc_ids)
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=5, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def _run_galago(self, func, arg_list):
        """
        run command line galago app
        :param func: either 'doc' or 'batch-search'
        :param arg_list:
        :return:
        """
        args = [self.galago_path, func, '--index=', self.index_path]
        args.extend(arg_list)
        p = subprocess.Popen(args, stdout=subprocess.PIPE)
        out, err = p.communicate()
        return out.decode("utf-8").strip()
