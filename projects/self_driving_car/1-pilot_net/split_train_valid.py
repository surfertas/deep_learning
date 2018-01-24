#!/usr/bin/env python
# @author Tasuku Miura
# @brief Utility for splitting data into train and validation.
#
# Script to split a given csv file into separate csv files for training and
# validation.
# 
# Note: You can set the defaults for original path and file name. Default for
# validation is set at 10% allocated to validation.

import argparse
import os

def main(args):
    with open(os.path.join(args.i, args.file_name), 'r') as f:
        csv_file= f.readlines()
        data_size = len(csv_file)
        valid_size = int(data_size * args.valid_split)

    with open(os.path.join(args.o, "valid_" + args.file_name), 'w+') as f:
        valid_file = f.writelines(csv_file[:valid_size])

    with open(os.path.join(args.o,"train_" + args.file_name), 'w+') as f:
        train_file = f.writelines(csv_file[valid_size:])


if __name__ == "__main__":
    default_original_path = default_out_path = r'/home/ubuntu/self_drive/data/output/bag_4'
    default_file_name = r'interpolated.csv'

    parser = argparse.ArgumentParser(
        description='Split into training and validation dataset')
    parser.add_argument('--i', type=str,
                        default=default_original_path,
                        help='Path to original dataset information')
    parser.add_argument('--o', type=str,
                        default=default_out_path,
                        help='Path to write train and valid dataset information')
    parser.add_argument('--file_name', type=str,
                        default=default_file_name,
                        help='File name that contains original data')
    parser.add_argument('--valid_split', type=float,
                        default=0.1,
                        help='Percentage to use as validation set')
    args = parser.parse_args()

    main(args)
