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

4. Run
```
$ python generate_csv_with_gcsurl.py
