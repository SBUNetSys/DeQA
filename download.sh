#!/bin/bash
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

# get corenlp
CORENLP_TARFILE="corenlp.tgz"
echo "Downloading corenlp library..."
echo
wget "${BASE_URL}/${CORENLP_TARFILE}"
tar -xvf ${CORENLP_TARFILE}
rm ${CORENLP_TARFILE}
echo

# get wikipedia data
WIKI_DIR="wikipedia"
mkdir -p ${WIKI_DIR}
cd ${WIKI_DIR}
for i in {a..d}
do
    echo "Downloading wikidata.tar.gz.${i}..."
    wget -O "wikidata.tar.gz.${i}" "${BASE_URL}/wikidata.tar.gz.${i}"
done

echo "Combining wikidata.tar.gz data slices into one..."
echo
cat `ls wikidata.tar.gz.*` > "wikidata.tar.gz"
tar -xvf "wikidata.tar.gz"
rm wikidata.tar.gz*

echo "DrQA download done!"
