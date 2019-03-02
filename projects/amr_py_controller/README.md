

1. create and activate conda environment and install necessary packages

```
$ conda env create -f environment.yml python=2.7
$ source activate amr-core
```


2. Create service key and define path.
```
$ export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/sa_key.json
```


- Copy folder with raw images and pickle file to google cloud. Initialize gcloud init. Use `gsutil ls`gs://<path_to_image_dir> > data/csv/gcs.csv`.
- Run generate_csv_with_gcsurl.py

4. Train. Make sure to specify "build_model_version" in the params file as that will direct which build model method to use in the `model/model_fn.py`
```
$ python train.py --model_dir ./experiments/resnet_v2_50/ --data_dir ./data/resnet_v2_50/
```
