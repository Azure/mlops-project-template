# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse

import numpy as np
import pandas as pd
import mlflow


def main(raw_data_path, prepared_data_path):

    print(f'Raw data path: {raw_data_path}')
    print(f'Output data path: {prepared_data_path}')

    # Read data

    labels_data = pd.read_csv(os.path.join(raw_data_path, 'image_labels.csv'))

    mlflow.log_metric('total_labels', len(labels_data))

    # Split data into train and test datasets

    random_data = np.random.rand(len(labels_data))
    labels_train = labels_data[random_data < 0.7]
    labels_test = labels_data[random_data >= 0.7]

    print(labels_train)

    mlflow.log_metric('train_size', labels_train.shape[0])
    mlflow.log_metric('test_size', labels_test.shape[0])

    labels_train.to_csv(os.path.join(prepared_data_path, 'labels_train.csv'), index=False)
    labels_test.to_csv(os.path.join(prepared_data_path, 'labels_test.csv'), index=False)

    print('Finished.')


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dogs-labels", type=str, required=True, help="Path to labels")
    parser.add_argument("--prepared_data_path", type=str, required=True, help="Path for prepared data")

    args_parsed, unknown = parser.parse_known_args(args_list)
    if unknown:
        print(f"Unrecognized arguments. These won't be used: {unknown}")

    return args_parsed


if __name__ == "__main__":
    args = parse_args()

    main(
        raw_data_path=args.dogs_labels,
        prepared_data_path=args.prepared_data_path
    )
