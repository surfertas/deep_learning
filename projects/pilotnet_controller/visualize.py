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


def visualize(model, backward_model, dataloader, cfg):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
    """

    # set model to evaluation mode
    model.eval()
    backward_model.eval()

    count = 0
    for data_batch, targets_batch in dataloader:

        indices = targets_batch.view(-1).nonzero().type(torch.LongTensor)
        # move to GPU if available
        if torch.cuda.is_available():
            data_batch, targets_batch = data_batch.cuda(async=True), targets_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, targets_batch = Variable(data_batch), Variable(targets_batch)

        # compute model output
        output_batch, activations_batch = model(data_batch, targets_batch)

        backward_segmentations = backward_model(activations_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        targets_batch = targets_batch.data.cpu().numpy()
        backward_segmentations = backward_segmentations.cpu().numpy()
        indices = indices.cpu()

        for i in indices:
            image_mask = backward_segmentations[i, :, :, :]
            path = cfg.OUTPUT.VIS_DIR + "/" + count
            save_image(image_mask, path + '_backward_segmentation.jpg')
            count += 1

    logging.info("- visualization done.")


if __name__ == '__main__':
    """
        Visualize the test set.
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
    utils.set_logger(os.path.join(args.model_dir, 'visualize.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, args.csv_filename, cfg)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = build_model(cfg, visualizing=True)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # Define VBP model
    backward_model = build_backward_model(cfg)
    backward_model.to(device)
    
    logging.info("Starting visualization")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # visualize
    visualize(model, backward_model, test_dl, cfg)

