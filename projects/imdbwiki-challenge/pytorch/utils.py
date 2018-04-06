# @author Tasuku Miura
# @brief Utils used for PyTorch training
import os
import torch


def save_checkpoint(state, file_name='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    print ("=> Saving a new best")
    torch.save(state, file_name)  # save checkpoint


def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
