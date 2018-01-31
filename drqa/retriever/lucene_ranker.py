#!/usr/bin/env python3

import logging
import subprocess
from multiprocessing.pool import ThreadPool
from functools import partial
import os
from pathlib import PosixPath
from drqa import tokenizers
from drqa.retriever import utils

logger = logging.getLogger(__name__)
TEXT_FLAG = '|TEXT:===>|'
DATA_DIR = (os.path.join(PosixPath(__file__).absolute().parents[2].as_posix(), 'data'))
DEFAULTS = {
    'lucene_path': os.path.join(DATA_DIR, 'LuceneTrecEnWiki-0.2/bin/LuceneTrecEnWiki'),
    'lucene_index': os.path.join(DATA_DIR, 'wikipedia/enwiki-lucene-20171103/')
}


class LuceneRanker(object):
    """Use Galago search engine to retrieve wikipedia articles
    """

    def __init__(self, lucene_path=None, index_path=None, use_keyword=True):
        self.question = None
        self.lucene_path = lucene_path or DEFAULTS['lucene_path']
        self.index_path = index_path or DEFAULTS['lucene_index']
        self.tokenizer = tokenizers.get_class('simple')()

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=1, uncased=True, filter_fn=utils.filter_ngram)

    def closest_docs(self, question_, k=5):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        doc_scores = []
        doc_ids = []
        doc_texts = []
        words = self.parse(utils.normalize(question_))
        query = ' '.join(words)
        if query.strip() is None:
            logger.warning('has no query!')
            return doc_ids, doc_scores, doc_texts
        search_results = self._run_lucene(query, k)
        for result in search_results.split('\n'):

            result_elements = result.split(TEXT_FLAG)
            # print(result_elements)
            id_and_score = result_elements[0].split()
            if len(id_and_score) < 2:
                logger.warning('query failed for question: %s' % question_)
                continue
            doc_id = id_and_score[0]
            doc_score = id_and_score[1]
            text = result_elements[1].strip()

            doc_ids.append(doc_id)
            doc_scores.append(doc_score)
            doc_texts.append(text)
            # print('id:', doc_id, 'ds:', doc_score, 'text:', text)
        logger.debug('question_d:%s, query:%s, doc_ids:%s, doc_scores:%s'
                     % (question_, query, doc_ids, doc_scores))
        return doc_ids, doc_scores, doc_texts

    def batch_closest_docs(self, queries, k=5, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def _run_lucene(self, query, top_n):
        """
        run command line lucene app
        :param query:
        :return:
        """
        args = [self.lucene_path, 'search', self.index_path, query, '%d' % top_n]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err:
            logger.warning('lucene question:%s args:[%s] error: ' % (self.question, args))
            logger.warning(err)
        return out.decode("utf-8").strip()


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    ranker = LuceneRanker()
    question = 'What sports stadium has been billed as "the eighth wonder of the world"?'
    ids, scores, texts = ranker.closest_docs(question, k=10)
    for doc_id, doc_score, doc_text in zip(ids, scores, texts):
        print(doc_id, doc_score, doc_text)

