"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import data.data_loader as data_loader
import model.all_metrics as all_metrics
from model import build_model, build_backward_model
from evaluate import evaluate
from config import get_cfg_defaults

parser = argparse.ArgumentParser()
parser.add_argument('--gcs_bucket_name', default='track-v0-night', help="GCS Bucket name where data is stored")
parser.add_argument('--data_dir', default='data/gcs', help="Directory containing the csv file")
parser.add_argument('--csv_filename', default='path_to_data.csv')
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, dataloader, metrics, cfg):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_targets and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, targets_batch) in enumerate(dataloader):
            # move to GPU if available
            if torch.cuda.is_available():
                train_batch, targets_batch = train_batch.cuda(async=True), targets_batch.cuda(async=True)

            # convert to torch Variables
            train_batch, targets_batch = Variable(train_batch), Variable(targets_batch)

            # compute model output and loss
            output_batch = model(train_batch)
#            print("output: {}".format(output_batch))
#            print("target: {}".format(targets_batch))
            
            loss = loss_fn(output_batch, targets_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % cfg.LOG.PERIOD == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                targets_batch = targets_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, targets_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
        metrics,
        cfg,
        model_dir,
        restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and targets of each batch
        cfg: parameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_rmse = np.inf

    for epoch in range(cfg.SOLVER.EPOCHS):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, cfg.SOLVER.EPOCHS))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, cfg)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, cfg)

        val_rmse = val_metrics['rmse']
        is_best = val_rmse <= best_val_rmse

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best rmse")
            best_val_rmse = val_rmse

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    #json_path = os.path.join(args.model_dir, 'params.json')
    #assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    #params = utils.Params(json_path)


    # Set the random seed for reproducible experiments
    torch.manual_seed(230)

    # use GPU if available
    if torch.cuda.is_available(): torch.cuda.manual_seed(230)

    cfg = get_cfg_defaults()
    cfg.freeze()

    # Set the logger
    utils.set_logger(os.path.join(cfg.LOG.PATH, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val', 'test'], args.gcs_bucket_name, args.data_dir, args.csv_filename, cfg)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model and optimizer
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)

    # fetch loss function and metrics
    loss_fn = torch.nn.MSELoss()

    metrics = all_metrics.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(cfg.SOLVER.EPOCHS))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, cfg, args.model_dir,
                       args.restore_file)
