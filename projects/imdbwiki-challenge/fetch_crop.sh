#!/usr/bin/env bash
# Script to download and untar the 7GB face only data and associated meta data
# found at https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

## tools
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
  local dir=${1}
  local file=${2}
  local url=${3}

  local fetch_flags=
  if [ $resume -ne 0 ]; then
    fetch_flags=-c
  fi
 
  mkdir -p ${outdir}/${dir} 
  ${FETCH} ${fetch_flags} -O ${outdir}/${dir}/${file} ${url}
}

function unwrap() {
  local dir=${1}
  local file=${2}
  local path_to_tar=${outdir}/${dir}

  ${UNZIP} -xvf ${path_to_tar}/$file -C ${path_to_tar}/
  echo "File has been placed in ${path_to_tar}"
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
# download dir file_name url
# uncomment to run

download imdb_crop imdb_crop.tar ${base_url}/imdb_crop.tar
#download meta imdb_meta.tar ${base_url}/imdb_meta.tar
unwrap imdb_crop imdb_crop.tar
#unwrap meta imdb_meta.tar 
