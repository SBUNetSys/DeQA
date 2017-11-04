#!/usr/bin/env python3


import logging
import subprocess

from multiprocessing.pool import ThreadPool
from functools import partial
from . import DEFAULTS

logger = logging.getLogger(__name__)


class GalagoDB(object):
    """Use Galago search engine to retrieve wikipedia articles
    """

    def __init__(self, galago_path=None, index_path=None):
        self.galago_path = galago_path or DEFAULTS['galago_path']
        self.index_path = index_path or DEFAULTS['galago_index']

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        args = ['--id=', doc_id]
        doc = self._run_galago('doc', args)
        start_pos = doc.find('<TEXT>')
        end_pos = doc.find('</TEXT>')
        text = doc[start_pos: end_pos].strip()
        # print(text)
        return text

    def close(self):
        pass

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
        # print(out.decode("utf-8").strip())
        return out.decode("utf-8").strip()
