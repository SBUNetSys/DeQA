#!/usr/bin/env python3
import argparse
import logging
from multiprocessing.pool import ThreadPool
from functools import partial
import os
from pathlib import PosixPath
from drqa import tokenizers
from drqa.retriever import utils
import lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher

logger = logging.getLogger(__name__)
DATA_DIR = (os.path.join(PosixPath(__file__).absolute().parents[2].as_posix(), 'data'))
DEFAULTS = {
    'lucene_index': os.path.join(DATA_DIR, 'wikipedia/enwiki-lucene/')
}


class LuceneSearcher(object):
    """Use Galago search engine to retrieve wikipedia articles
    """

    def __init__(self, index_path=None):
        self.question = None
        self.index_path = index_path or DEFAULTS['lucene_index']
        self.tokenizer = tokenizers.get_class('simple')()
        self.env = lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        directory = SimpleFSDirectory(Paths.get(self.index_path))
        self.analyzer = StandardAnalyzer()
        # self.query_parser = MultiFieldQueryParser(["title", "text"], self.analyzer)

        self.searcher = IndexSearcher(DirectoryReader.open(directory))

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
        if not query:
            logger.warning('has no query!')
            return doc_ids, doc_scores, doc_texts

        # bq_builder = BooleanQuery.Builder()
        # title_query = TermQuery(Term("title", query))
        # # boosted_title_query = BoostQuery(title_query, 2)
        # bq_builder.add(TermQuery(Term("text", query)), BooleanClause.Occur.SHOULD)
        # bq_builder.add(title_query, BooleanClause.Occur.SHOULD)
        # lucene_query = bq_builder.build()

        # lucene_query = self.query_parser.parse(query, ["title", "text"],
        #                                        [BooleanClause.Occur.SHOULD, BooleanClause.Occur.MUST],
        #                                        self.analyzer)
        # lucene_query = 'title:"{0}"^2 OR "{0}"'.format(query)

        self.env.attachCurrentThread()
        query_parser = QueryParser("text", self.analyzer)
        search_results = self.searcher.search(query_parser.parse(query), k).scoreDocs
        for search_result in search_results:
            doc = self.searcher.doc(search_result.doc)
            doc_id = doc["id"] + ", title=" + doc["title"]
            doc_score = search_result.score
            text = doc["text"]
            doc_ids.append(doc_id)
            doc_scores.append(doc_score)
            doc_texts.append(text)
            # print('id:', doc_id, 'ds:', doc_score, 'text:', text)
        # logger.debug('question_d:%s, query:%s, doc_ids:%s, doc_scores:%s'
        #              % (question_, query, doc_ids, doc_scores))
        return doc_ids, doc_scores, doc_texts

    def batch_closest_docs(self, queries, k=5, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results


def main(args):
    ranker = LuceneSearcher(args.lucene_index)
    question = args.question
    ids, scores, texts = ranker.closest_docs(question, k=150)
    for i, (doc_id, doc_score, doc_text) in enumerate(zip(ids, scores, texts), 1):
        print('{}, id={}, score={:.3f}'.format(i, doc_id, doc_score))
        print(doc_text)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--lucene_index', type=str,
                        default='/Volumes/HDD/data/DrQA_data/data/wikipedia/enwiki-lucene')
    parser.add_argument('-q', '--question', type=str, default='How tall is Mount McKinley?')
    main(parser.parse_args())
