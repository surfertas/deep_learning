## Scripts related to processing of IMDB data and training of CNN model.
---


data source:
[https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

paper: [DEX: Deep EXpectation of apparent age from a single
image](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01229.pdf)

related blog post:[surfertas notes](http://surfertas.github.io/deeplearning/2017/04/18/imdbwiki.html)

Note: Completed as part of [Idein](https://idein.jp/) Residence Program


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

### Usage
To see sample usage, just git clone and run below.
```bash
python detect_age.py --file_name imdb_data_50.pkl --gpu 0
```

If you want to generate own data set, and set up environment for
experimentation use the following.
```bash
# downloads the relevant data from imdb-wiki site
./fetch_crop.sh

# preprocesses the raw data using 1000 samples
python imdb_preprocess.py --partial 1000

# initiate training
python detect_age.py --file_name imdb_data_1000.pkl --gpu 0 --white True
```
