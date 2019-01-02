

1. create and activate conda environment and install necessary packages

```
$ conda env create -f environment.yml python=3.6
$ source activate amr-core
```


2. Create service key and define path.
```
$ export GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/sa_key.json
```

3. Copy folder with raw images and pickle file to google cloud. Use `gsutil ls` and place results in `data/csv/gcs.csv`.


2. Got to `/experiments` and create a directory for your experiment. Create a params.json file and place in directory. (Check other experiments for inspiration.). Make sure to specify "experiment_name" as that will be used as the sub directory path to store the TFRecord files. (e.g. "experiment_name": "resnet_v2_50" -> `data/resnet_v2_50). Ideally name the experiments directory you will be using to the same name, e.g `/experiments/resnet_v2_50)

3. Build the tfrecord dataset. Specify "dataset_size" in params.json file. Also make sure to specify the required sshape, "num_channels", and "image_size" for your input to the architecture you plan to use as the build process will resize the images appropriate for the given model. 

```
$ python build_tfrecord_dataset.py --model_dir ./experiments/resnet_v2_50/
```
4. Check what devices are available. (When you are unsure if GPUs are available~
```
$ python check_devices.py
```

5. Train. Make sure to specify "build_model_version" in the params file as that will direct which build model method to use in the `model/model_fn.py`
```
$ python train.py --model_dir ./experiments/resnet_v2_50/ --data_dir ./data/resnet_v2_50/
```
