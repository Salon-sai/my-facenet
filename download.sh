#!/bin/bash
mkdir -p data
cd data

if [ ! -f lfw.tgz ]; then
    wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
fi

if [ ! -d imdb_crop ]; then
    tar zxvf lfw.tgz
fi

if [ ! -f imdb_crop.tar ]; then
    wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
fi

if [ ! -d imdb_crop ]; then
    tar xf imdb_crop.tar
fi

if [ ! -f wiki_crop.tar ]; then
    wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
fi

if [ ! -d wiki_crop ]; then
    tar xf wiki_crop.tar
fi