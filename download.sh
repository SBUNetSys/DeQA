#!/bin/bash
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

# Configure download location
DOWNLOAD_PATH="$DRQA_DATA"
if [ "$DRQA_DATA" == "" ]; then
    echo "DRQA_DATA not set; downloading to default path ('data')."
    DOWNLOAD_PATH="./data"
fi

mkdir -p ${DOWNLOAD_PATH}

cd ${DOWNLOAD_PATH}

BASE_URL="https://github.com/csarron/data/releases/download/v0.3"

# get qa data
QADATA_TARFILE="datasets.tgz"
echo "Downloading qa dataset..."
echo
wget "${BASE_URL}/${QADATA_TARFILE}"
tar -xvf ${QADATA_TARFILE}
rm ${QADATA_TARFILE}

# get reader model
READER_DIR="reader"
mkdir -p ${READER_DIR}
echo "Downloading reader models..."
echo
wget -O "${READER_DIR}/multitask.mdl" "${BASE_URL}/multitask.mdl"
wget -O "${READER_DIR}/single.mdl" "${BASE_URL}/single.mdl"

# get tfidf data
WIKI_DIR="wikipedia"
mkdir -p ${WIKI_DIR}
echo "Downloading tfidf-matrix data slices..."
echo
wget -O "${WIKI_DIR}/tfidf-matrix.npz.1" "${BASE_URL}/tfidf-matrix.npz.1"
wget -O "${WIKI_DIR}/tfidf-matrix.npz.2" "${BASE_URL}/tfidf-matrix.npz.2"

echo "Combining tfidf-matrix data slices into one..."
echo
cat `ls ${WIKI_DIR}/tfidf-matrix.npz.*` > "${WIKI_DIR}/tfidf-matrix.npz"
rm ${WIKI_DIR}/tfidf-matrix.npz.*

echo "Downloading tfidf-meta data..."
echo
wget -O "${WIKI_DIR}/tfidf-meta.npz" "${BASE_URL}/tfidf-meta.npz"

# get wikipedia data
for i in {a..g}
do
    echo "Downloading docs.db.${i}..."
    wget -O "${WIKI_DIR}/docs.db.${i}" "${BASE_URL}/docs.db.${i}"
done

echo "Combining docs.db data slices into one..."
echo
cat `ls ${WIKI_DIR}/docs.db.*` > "${WIKI_DIR}/docs.db"
rm ${WIKI_DIR}/docs.db.*


# get corenlp
CORENLP_TARFILE="corenlp.tgz"
echo "Downloading corenlp library..."
echo
wget "${BASE_URL}/${CORENLP_TARFILE}"
tar -xvf ${CORENLP_TARFILE}
rm ${CORENLP_TARFILE}
echo

echo "DrQA download done!"
