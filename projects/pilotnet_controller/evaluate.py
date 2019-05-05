"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.all_metrics as all_metrics
import data.data_loader as data_loader
from model import build_model, build_backward_model
from config import get_cfg_defaults

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/gcs', help="Directory containing the dataset")
parser.add_argument('--csv_filename', default='path_to_data2059.csv')
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, cfg, test=False):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    outputs = np.array([])
    targets = np.array([])
    # compute metrics over the dataset
    for data_batch, targets_batch in dataloader:

        # move to GPU if available
        if torch.cuda.is_available():
            data_batch, targets_batch = data_batch.cuda(async=True), targets_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, targets_batch = Variable(data_batch), Variable(targets_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, targets_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        targets_batch = targets_batch.data.cpu().numpy()


        # construct for visualization in test evaluation
        outputs = np.append(outputs, output_batch)
        targets = np.append(targets, targets_batch)

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, targets_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data.item()
        summ.append(summary_batch)

    if test:
        plt.plot(outputs)
        plt.plot(targets)
        plt.savefig('steering')

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()

    # Set the random seed for reproducible experiments
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    # use GPU if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(230)

    cfg = get_cfg_defaults()
    cfg.freeze()

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, args.csv_filename, cfg)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    loss_fn = torch.nn.MSELoss()
    metrics = all_metrics.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, cfg, test=True)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
