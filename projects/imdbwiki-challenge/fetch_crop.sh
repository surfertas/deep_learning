#!/usr/bin/env bash

## TOOLs
FETCH=wget
UNZIP=tar

## base url
base_url=https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static

## default settings
# output directory
outdir=.
# resume download
resume=0


function usage_exit () {
  echo "Usage: $0 [-c] [-h] [-o outdir]"
  echo ""
  echo "DESCRIPTION:"
  echo "  Download IMDB cropped images dataset."
  echo ""
  echo "OPTIONS:"
  echo "  -c          continue getting a partially-downloaded file"
  echo "  -h          print this message"
  echo "  -o outdir   output directory(default: $outdir)"
  echo ""
  echo "NOTICE:"
  echo "  This dataset is only the cropped data, faces only."
  echo ""
  exit 1
}

function download () {
  local out=${1}
  local url=${2}

  local fetch_flags=
  if [ $resume -ne 0 ]; then
    fetch_flags=-c
  fi
  
  ${FETCH} ${fetch_flags} -O ${outdir}/${out} ${url}
}

function unwrap() {
  local file=${1}
  local dir=${2}

  ${UNZIP} -xvf $file
  mv ./$dir/* ./data 
  rm -r $dir
  rm $file
  echo "Results placed in ./data"
}

## parse options
while getopts cdho: OPT
do
  case $OPT in
    h) usage_exit
       ;;
    c) resume=1
       ;;
    o) outdir=$OPTARG
       ;;
    *) echo "Unknown option ($OPT)"
       usage_exit
       ;;
  esac
done

## IMDB
# fetch imdb cropped dataset and meta
#download imdb_crop.tar ${base_url}/imdb_crop.tar
download imdb_meta.tar ${base_url}/imdb_meta.tar
unwrap imdb_meta.tar imdb
#unwrap imdb_crop.tar imdb_crop
