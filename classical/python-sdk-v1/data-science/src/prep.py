# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from azureml.core import Run

import argparse

run = Run.get_context()
ws = run.experiment.workspace

def parse_args():
    parser = argparse.ArgumentParser(description="UCI Credit example")
    parser.add_argument("--uci-credit", type=str, default='data/', help="Directory path to training data")
    parser.add_argument("--prepared_data_path", type=str, default='prepared_data/', help="prepared data directory")
    parser.add_argument("--enable_monitoring", type=str, default="false", help="enable logging to ADX")
    parser.add_argument("--table_name", type=str, default="mlmonitoring", help="Table name in ADX for logging")
    return parser.parse_known_args()

def log_training_data(df, table_name):
    from obs.collector import Online_Collector
    from datetime import timedelta
    print("If there is an Authorization error, check your Azure KeyVault secret named kvmonitoringspkey. Terraform might put single quotation marks around the secret. Remove the single quotes and the secret should work.")
    collector = Online_Collector(table_name)
    df["timestamp"] = [pd.to_datetime('now') - timedelta(days=x) for x in range(len(df))]
    collector.batch_collect(df)


def main():
    # Parse command-line arguments
    args, unknown = parse_args()
    prepared_data_path = args.prepared_data_path

    # Make sure data output path exists
    if not os.path.exists(prepared_data_path):
        os.makedirs(prepared_data_path)
        
    # Enable auto logging
    mlflow.sklearn.autolog()
    
    # Read training data
    df = pd.read_csv(os.path.join(args.uci_credit, 'credit.csv'))

    random_data = np.random.rand(len(df))

    msk_train = random_data < 0.7
    msk_val = (random_data >= 0.7) & (random_data < 0.85)
    msk_test = random_data >= 0.85
    
    train = df[msk_train]
    val = df[msk_val]
    test = df[msk_test] 
    
    run.log('TRAIN SIZE', train.shape[0])
    run.log('VAL SIZE', val.shape[0])
    run.log('TEST SIZE', test.shape[0])
    
    run.parent.log('TRAIN SIZE', train.shape[0])
    run.parent.log('VAL SIZE', val.shape[0])
    run.parent.log('TEST SIZE', test.shape[0])
    
    TRAIN_PATH = os.path.join(prepared_data_path, "train.csv")
    VAL_PATH = os.path.join(prepared_data_path, "val.csv")
    TEST_PATH = os.path.join(prepared_data_path, "test.csv")
    
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    if (args.enable_monitoring.lower() == 'true' or args.enable_monitoring == '1' or args.enable_monitoring.lower() == 'yes'):
        log_training_data(df, args.table_name)
    
if __name__ == '__main__':
    main()
