#!/usr/bin/env python3

import logging
import subprocess

from multiprocessing.pool import ThreadPool
from functools import partial
from . import utils
from . import DEFAULTS
from .. import tokenizers
from .rake import Rake

logger = logging.getLogger(__name__)


class GalagoRanker(object):
    """Use Galago search engine to retrieve wikipedia articles
    """

    def __init__(self, galago_path=None, index_path=None, use_keyword=True):
        self.galago_path = galago_path or DEFAULTS['galago_path']
        self.index_path = index_path or DEFAULTS['galago_index']
        self.use_keyword = use_keyword
        self.tokenizer = tokenizers.get_class('simple')()

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=2, uncased=True, filter_fn=utils.filter_ngram)

    def closest_docs(self, question, k=5):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        if self.use_keyword:
            keyword_items = Rake().run(question)
            word_queries = []
            for keyword_item in keyword_items:
                keyword, _ = keyword_item
                # keyword_query = '#od:1( %s )' % keyword if ' ' in keyword else keyword
                keyword_query = keyword.replace('-', ' ')
                word_queries.append(keyword_query)
        else:
            word_queries = self.parse(utils.normalize(question))

        query = ' '.join(word_queries)
        args = ['--requested=%s' % k, '--casefold=true', '--query=', '#combine(%s)' % query]
        search_results = self._run_galago('batch-search', args)
        doc_scores = []
        doc_ids = []
        doc_texts = []
        for result in search_results.split('</NE>'):
            if not result:
                continue
            result_elements = result.split('<TEXT>')
            # print(result_elements)
            meta_info_list = result_elements[0].split()
            if len(meta_info_list) < 7:
                print('query failed')
                continue
            doc_id = meta_info_list[2]
            doc_score = meta_info_list[4]

            end_pos = result_elements[1].find('</TEXT>')
            text = result_elements[1][0: end_pos].strip()

            doc_ids.append(doc_id)
            doc_scores.append(doc_score)
            doc_texts.append(text)
            # print('doc_id:', doc_id, 'doc_text:', text)
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
        if err:
            logger.warning(err)
        return out.decode("utf-8").strip()
