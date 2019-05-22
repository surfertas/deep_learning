import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', help="Path to csv file")


def sample_data(csv_path: str, every_n: int):
    """ Sample data by taking every nth row. """
    df = pd.read_csv(csv_path)
    df_sampled = df.iloc[::every_n]
    return df_sampled


if __name__ == "__main__":
    args = parser.parse_args()
    df = sample_data(
        csv_path=args.csv_path,
        every_n=3)

    df.to_csv("path_to_sampled.csv")
