#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from abc import abstractmethod
from datetime import datetime

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document
from org.apache.lucene.document import Field
from org.apache.lucene.document import StringField
from org.apache.lucene.document import TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import NIOFSDirectory

from eqa import logger
from eqa.util.text import word_tokenize, rm_white_space, rm_special_chars


class Indexer(object):

    def __init__(self, index_store_path):

        store = NIOFSDirectory(Paths.get(index_store_path))
        analyzer = StandardAnalyzer()
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
        self.writer = IndexWriter(store, config)

    @abstractmethod
    def index_single_file(self, doc_file):
        pass

    def index_doc(self, doc_path):
        if os.path.isfile(doc_path):
            return 1, self.index_single_file(doc_path)

        # index all docs in doc_path dir
        total = 0
        doc_num = 0
        for root, _, files in os.walk(doc_path, topdown=False):
            for name in files:
                doc_file = os.path.join(root, name)
                total += self.index_single_file(doc_file)
                doc_num += 1
        return doc_num, total

    def __del__(self):
        logger.info('committing index...')
        self.writer.commit()
        self.writer.close()
        logger.info('done')


class ParaIndexer(Indexer):
    def __init__(self, index_store_path):
        super(ParaIndexer, self).__init__(index_store_path)

    def index_single_file(self, doc_file):
        logger.info("adding {}".format(doc_file))

        single_file_num = 0
        try:
            with open(doc_file) as df:
                for line in df:
                    para_no = 1
                    wiki_doc = json.loads(line)
                    doc_title = wiki_doc['title']
                    doc_text = wiki_doc['plaintext']
                    doc_id = wiki_doc['_id']
                    paragraphs = doc_text.split('\n\n')
                    if len(paragraphs) < 3:
                        continue
                    # logger.info('doc_id:', doc_id, 'title:', doc_title, 'para_num:', len(paragraphs))
                    for para in paragraphs:
                        para = rm_white_space(para)
                        if len(word_tokenize(para)) < 50:
                            continue
                        para_id = '{}_{}'.format(doc_id, para_no)
                        doc = Document()
                        doc.add(StringField("id", para_id, Field.Store.YES))
                        doc.add(TextField("title", doc_title, Field.Store.YES))
                        doc.add(TextField("text", para, Field.Store.YES))
                        self.writer.addDocument(doc)
                        para_no += 1
                        single_file_num += 1
                        if single_file_num % 10000 == 0:
                            logger.info('added {} lucene docs (paragraphs)'.format(single_file_num))
        except Exception as e:
            import traceback
            traceback.print_tb(e.__traceback__)
            logger.error("Failed in: {}".format(doc_file))

        return single_file_num


class DocIndexer(Indexer):
    def __init__(self, index_store_path):
        super(DocIndexer, self).__init__(index_store_path)

    def index_single_file(self, doc_file):
        logger.info("adding {}".format(doc_file))
        lucene_doc_num = 0
        try:
            with open(doc_file) as df:
                for line in df:
                    wiki_doc = json.loads(line)
                    doc_title = wiki_doc['title']
                    doc_text = wiki_doc['plaintext']
                    doc_id = wiki_doc['_id']
                    paragraphs = doc_text.split('\n\n')
                    if len(paragraphs) < 3:
                        continue
                    doc_text = rm_special_chars(doc_text)
                    doc = Document()
                    doc.add(StringField("id", str(doc_id), Field.Store.YES))
                    doc.add(TextField("title", doc_title, Field.Store.YES))
                    doc.add(TextField("text", doc_text, Field.Store.YES))
                    self.writer.addDocument(doc)
                    lucene_doc_num += 1
                    if lucene_doc_num % 10000 == 0:
                        logger.info('added {} lucene docs'.format(lucene_doc_num))
        except Exception as e:
            import traceback
            traceback.print_tb(e.__traceback__)
            logger.error("Failed in: {}".format(doc_file))
        return lucene_doc_num


def main(args):
    doc_path = args.doc_path
    if not os.path.exists(doc_path):
        raise ValueError('{} not exist!'.format(doc_path))
    index_dir = args.index_dir
    os.makedirs(index_dir, exist_ok=False)
    index_type = args.index_type

    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    logger.info('using lucene {}'.format(lucene.VERSION))
    logger.info('doc_path: {}, index_dir:{}, index_type: {}'.format(doc_path, index_dir, index_type))
    start = datetime.now()
    if index_type == 'para':
        indexer = ParaIndexer(index_dir)
    else:
        indexer = DocIndexer(index_dir)
    indexed_files_num, indexed_docs_num = indexer.index_doc(doc_path)
    end = datetime.now()
    logger.info("added {} files, {} docs, took {} s".format(indexed_files_num, indexed_docs_num, end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--doc_path', type=str, default=None)
    parser.add_argument('-i', '--index_dir', type=str, default=None)
    parser.add_argument('-it', '--index_type', choices=('para', 'doc'), default='para')

    main(parser.parse_args())
