import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', help="Path to csv file")


def balance_data(csv_path: str, thresh: float, frac: float):
    """ Balances data by reducing the number of "driving straight" samples """
    df = pd.read_csv(csv_path)
    df_angled = df[np.abs(df['steer']) > thresh]
    df_nangled = df[np.abs(df['steer']) < thresh].sample(frac=frac)
    df_return = pd.concat([df_angled, df_nangled], axis=0)
    return df_return


if __name__ == "__main__":
    args = parser.parse_args()
    df = balance_data(
        csv_path=args.csv_path,
        thresh=0.1,
        frac=0.3)

    df.to_csv("path_to_data_balanced.csv")
