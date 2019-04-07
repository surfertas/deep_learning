### WIP

1. create and activate conda environment and install necessary packages

```
$ conda env create -f environment.yml python=2.7
$ source activate amr-core
```


2. Create service key for use with GCP and place in sa_key.json. Run the start up script.
```
$ ./startup.sh
```
- Copy folder with raw images and pickle file to google cloud. Initialize gcloud init. 
- Create the a csv file with the paths to images. Use `gsutil ls`gs://<path_to_image_dir> > data/csv/gcs.csv.
- Run generate_csv_with_gcsurl.py

4. Train. Make sure to specify "build_model_version" in the params file as that will direct which build model method to use in the `model/model_fn.py`
```
$ python train.py --model_dir ./experiments/resnet_v2_50/ --data_dir ./data/resnet_v2_50/
```