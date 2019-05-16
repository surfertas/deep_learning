# PilotNet Controller

A PyTorch research environment for training PilotNet to predict control (steering) for a RC car. 

Image features and controller targets are collected using a RC car(Donkey) running the [amr_core]() software.

The data stored on the SSD is transferred to storage on Google Storage, and then transfered to an GCP instance with a GPU.

## Setup
1. Create and activate the conda environment and install the necessary packages

```
$ conda env create -f environment.yml python=3.6
$ source activate amr-core
```

2. Once the environment has been set up, we need to generate a csv file that can be used with the training engine. Follow the instructions found [here](https://surfertas.github.io/amr/deeplearning/machinelearning/2019/03/31/amr-4.html) to get the images from SSD to the VM instance on GCP. Once complete you should have a `path_to_data.csv` file located int `/data/gcs`.

3. Review the configuration file found at `config/default.py` for modifiable configurations. Once you are satisfied you can kick off the trainiing by running the following command.

```
python train.py --data_dir --csv_filename --model_dir --restore_file 
```

4. Run base case analysis vs. a target of zero and a target using the mean.

```
python base_case_rmse.py --csv_path
```

## Acknowledgement
1. [cs230-code-examples](https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision)
2. [end2end-self-driving-car](https://github.com/MahanFathi/end2end-self-driving-car)

## License
[MIT](https://choosealicense.com/licenses/mit/)
