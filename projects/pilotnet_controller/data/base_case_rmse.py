# @date: 5/2/2019
import argparse
import pandas as pd
import numpy as np
from typing import Tuple

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', help="Path to csv file")

def base_case(csv_path: str) -> Tuple[float]:
    """ Computes the RMSE using mean and zero as input. """
    df = pd.read_csv(csv_path)
    n = df['steer'].count()
    rmse_mean = np.sqrt(1./n*np.sum(np.square(df['steer']-df['steer'].mean())))
    rmse_zero = np.sqrt(1./n*np.sum(np.square(df['steer'])))
    return rmse_mean, rmse_zero


if __name__=="__main__":
    args = parser.parse_args()
    rmse_mean, rmse_zero = base_case(args.csv_path)
    print("RMSE Mean: {}".format(rmse_mean))
    print("RMSE Zero: {}".format(rmse_zero))


