"""Train the model"""

import argparse
import logging
import os
import random

import tensorflow as tf

from model.input_fn import input_tf_records_all_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/raw-tfrecords',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    # model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    # overwritting = model_dir_has_best_weights and args.restore_from is None
    # assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")

    # Specify the sizes of the dataset we train on
    # train_ratio and dataset_size specified in paraams
    params.train_size = int(params.train_ratio * params.dataset_size)
    params.eval_size = params.dataset_size - params.train_size
    logging.info("Train size: {} Eval size: {} Total: {}".format(params.train_size, params.eval_size, params.dataset_size))
    # Create the two iterators over the two datasets
    train_inputs, eval_inputs = input_tf_records_all_fn(args.data_dir, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.dict['num_epochs']))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)
