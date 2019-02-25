

1. create and activate conda environment and install necessary packages

```
$ conda env create -f environment.yml python=2.7
$ source activate amr-core
```


2. Create service key and define path.
```
$ export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/sa_key.json
```

3. Copy folder with raw images and pickle file to google cloud. Use `gsutil ls` and place results in `data/csv/gcs.csv`.


4. Train. Make sure to specify "build_model_version" in the params file as that will direct which build model method to use in the `model/model_fn.py`
```
$ python train.py --model_dir ./experiments/resnet_v2_50/ --data_dir ./data/resnet_v2_50/
```
