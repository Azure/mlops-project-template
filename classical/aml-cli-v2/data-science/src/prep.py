import argparse

from pathlib import Path
import os
import numpy as np
import pandas as pd

import mlflow


def parse_args():

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--prepared_data", type=str, help="Path of prepared data")

    args = parser.parse_args()

    return args

def main():

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Data output path: {args.prepared_data}",
    ]

    for line in lines:
        print(line)

    # ------------ Reading Data ------------ #
    # -------------------------------------- #

    print("mounted_path files: ")
    arr = os.listdir(args.raw_data)
    print(arr)

    data = pd.read_csv((Path(args.raw_data) / 'taxi-data.csv'))

    # ------------- Split Data ------------- #
    # -------------------------------------- #

    # Split data into train, val and test datasets

    random_data = np.random.rand(len(data))

    msk_train = random_data < 0.7
    msk_val = (random_data >= 0.7) & (random_data < 0.85)
    msk_test = random_data >= 0.85

    train = data[msk_train]
    val = data[msk_val]
    test = data[msk_test] 

    mlflow.log_metric('train size', train.shape[0])
    mlflow.log_metric('val size', val.shape[0])
    mlflow.log_metric('test size', test.shape[0])

    train.to_csv((Path(args.prepared_data) / "train.csv"))
    val.to_csv((Path(args.prepared_data) / "val.csv"))
    test.to_csv((Path(args.prepared_data) / "test.csv"))

if __name__ == "__main__":
    main()