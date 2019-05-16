import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', help="Path to csv file")
parser.add_argument('--n_samples', help="Number of samples to visualize", type=int)


def get_visualize_data(csv_path: str, n: int):
    """ Balances data by reducing the number of "driving straight" samples """
    df = pd.read_csv(csv_path)
    df = df[:n]
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    df = get_visualize_data(
        csv_path=args.csv_path,
        n=args.n_samples)

    df.to_csv("./gcs/path_to_data_visualize.csv")
