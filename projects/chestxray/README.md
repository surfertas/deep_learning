<a href='https://www.korabo.io/checkout-page/5e2c4727d31a7a00290de877/5e23f0b31b70f70022ff5933'>
    <img src='https://img.shields.io/badge/Korabo-donate-blue' alt='donate'>
</a>


[Medical imaging: playing with the ChestXray-14 dataset](https://surfertas.github.io/deeplearning/2018/12/12/chestxray.html)

1. activate environment and install necessary packages

```
$ virtualenv -p python3 .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```

2. Got to `/experiments` and create a directory for your experiment. Create a params.json file and place in directory. (Check other experiments for inspiration.). Make sure to specify "experiment_name" as that will be used as the sub directory path to store the TFRecord files. (e.g. "experiment_name": "resnet_v2_50" -> `data/resnet_v2_50). Ideally name the experiments directory you will be using to the same name, e.g `/experiments/resnet_v2_50)

3. Build the tfrecord dataset. Specify "dataset_size" in params.json file. Also make sure to specify the required shape, "num_channels", and "image_size" for your input to the architecture you plan to use as the build process will resize the images appropriate for the given model. 

```
$ python build_tfrecord_dataset.py --model_dir ./experiments/resnet_v2_50/
```
4. Check what devices are available. (When you are unsure if GPUs are available)
```
$ python check_devices.py
```

5. Train. Make sure to specify "build_model_version" in the params file as that will direct which build model method to use in the `model/model_fn.py`
```
$ python train.py --model_dir ./experiments/resnet_v2_50/ --data_dir ./data/resnet_v2_50/
```
