#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
import argparse
import os

try:
    import urllib.request as urlrequest
except ImportError:
    import urllib as urlrequest


def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    import requests
    from tqdm import tqdm
    file_size = int(urlrequest.urlopen(url).info().get('Content-Length', -1))
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--date', type=str, default='20180120',
                        help='enwiki dumps date, e.g. 20180120')
    parser.add_argument('-p', '--path', type=str, default='data/',
                        help='dump path')
    args = parser.parse_args()
    base_url = 'http://dumps.wikimedia.your.org/enwiki/'
    wiki_url = '{0}{1}/enwiki-{1}-pages-articles.xml.bz2'.format(base_url, args.date)
    save_name = 'enwiki-{0}-pages-articles.xml.bz2'.format(args.date)
    print('downloading enwiki dumps from: {} to {}'.format(wiki_url, save_name))
    download_from_url(wiki_url, os.path.join(args.path, save_name))
