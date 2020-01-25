<a href='https://www.korabo.io/checkout-page/5e2c4727d31a7a00290de877/5e23f0b31b70f70022ff5933'>
    <img src='https://img.shields.io/badge/Korabo-donate-blue' alt='donate'>
</a>


## Scripts related to processing of IMDB data and training of CNN model.
---

data source:
[https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

paper: [DEX: Deep EXpectation of apparent age from a single
image](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01229.pdf)

related blog post:[surfertas notes](http://surfertas.github.io/deeplearning/2017/04/18/imdbwiki.html)

Note: Completed as part of [Idein](https://idein.jp/) Residence Program (Apr/2017)

Updated:
Apr/2018 - Fixed bugs, implemented sampling of data. Transition to Pytorch as
main implementation.

### Introduction

This is an attempt to construct an age classifier with limited resources. The
state of the art results obtained in "DEX: Deep EXpectation of apparent age from
a single image" was the result of a large data set, large capacity architecture,
and access to resources that would allow for 5 days worth of training.

The handling of such data (hundreds of GBs), and sufficient GPU access is
somewhat unreasonable to an individual. 

Thus this exercise is to see if we can answer the question of whether or not we
can generate a reasonable model that can find some practical use, using only a
subset of the training data, and a model with lower capacity relative to the
architecture introduced in the referenced paper.

The scripts to download and prepare the data hopefully will be useful for
others.

If you want to generate the original data set, and set up environment for
experimentation use the following.
```bash
# downloads the relevant data from imdb-wiki site
# NOTE: will only download the 7gb faces only data and associated meta file
$ ./fetch_crop.sh -o output_path

# preprocesses the raw data using 1000 samples
# NOTE: If using Chainer models and trainer need to set --get-paths False 
$ python imdb_preprocess.py --n-samples 1000

# initiate training using pytorch implementation
# MAKE SURE TO SPECIFY THE PATH as default is set for my personal use...
$ cd pytorch
$ python trainer.py --root-dir path_to_pickle --train-data imdb_data_1000.pkl
```
