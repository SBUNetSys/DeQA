#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.index import Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search import TermQuery
from org.apache.lucene.store import SimpleFSDirectory

from eqa import logger, INDEX_DIR


def main(args):
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    logger.info('lucene: {}'.format(lucene.VERSION))
    index = args.index or INDEX_DIR
    logger.info('index dir: {}'.format(index))
    query = args.query
    logger.info("getting doc for query={}".format(query))

    directory = SimpleFSDirectory(Paths.get(index))
    searcher = IndexSearcher(DirectoryReader.open(directory))
    analyzer = StandardAnalyzer()
    field = args.field
    if field == 'text':
        lucene_query = QueryParser("text", analyzer).parse(query)
    else:
        lucene_query = TermQuery(Term(field, query))
    score_docs = searcher.search(lucene_query, args.num).scoreDocs
    logger.info("%s total matching documents." % len(score_docs))
    for scoreDoc in score_docs:
        doc = searcher.doc(scoreDoc.doc)
        logger.info('id={}, title={}, score={:.6f}\ntext: {}\n'.format(doc.get("id"), doc.get("title"),
                                                                       scoreDoc.score, doc.get("text")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=str, default=None)
    parser.add_argument('-f', '--field', choices=('id', 'title', 'text'), default='title',
                        help="text field will use search instead of exact match")
    parser.add_argument('-q', '--query', type=str, default='Mount Everest')
    parser.add_argument('-n', '--num', type=int, default=10)

    main(parser.parse_args())
