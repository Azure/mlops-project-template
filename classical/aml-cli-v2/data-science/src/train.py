# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse
import os

import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from azure.identity import DefaultAzureCredential 
from azure.ai.ml.entities import AzureDataLakeGen2Datastore
from azure.ai.ml import MLClient

import mlflow
import mlflow.sklearn

import utils
from dotenv import load_dotenv
from pathlib import Path

def parse_args():
    '''Parse input arguments.'''
    parser = argparse.ArgumentParser()
    # parser.add_argument("--azure_client_id", type=str)
    # parser.add_argument("--azure_tenant_id", type=str)
    parser.add_argument("--data_path", type=str, help="Path to input data.")
    # parser.add_argument("--registered_model_name", type=str, help="Model name.")
    parser.add_argument("--config_path", type=str, help="Path to AML config file.")
    parser.add_argument("--model_output", type=str, help="Path to output model.")

    args = parser.parse_args()

    return args

def main(args):
    load_dotenv()

    # os.environ['AZURE_CLIENT_ID'] = args.azure_client_id
    # os.environ['AZURE_TENANT_ID'] = args.azure_tenant_id

    # ml_client = MLClient.from_config(DefaultAzureCredential(), path=args.config_path)
    # ws = ml_client.workspaces.get(ml_client.workspace_name) 

    df = pd.read_parquet(Path(args.data_path))
    print("Dataframe shape:", df.shape)
    # df = pd.read_csv(args.data_path, header=1, index_col=0)
    # # Create/retrieve the necessary clients and upload the data
    # adls_system_url = f"{utils.fs_config.get('adls_scheme')}://{utils.fs_config.get('adls_account')}.dfs.core.windows.net"
    # service_client = DataLakeServiceClient(
    #     account_url=adls_system_url, credential=os.environ['ADLS_KEY'])


    # file_system_client = utils.create_or_retrieve_file_system(service_client, utils.fs_config.get('adls_file_system'))
    # directory_client = utils.create_or_retrieve_directory(file_system_client, utils.fs_config.get('adls_data_directory'))
    # file_client = utils.create_or_retrieve_file(directory_client, utils.fs_config.get('adls_data_file'))
    # # Download the file
    # download = file_client.download_file()
    # downloaded_bytes = download.readall()
    # df = pd.read_csv(BytesIO(downloaded_bytes))

    # Start Logging
    # mlflow.set_tracking_uri(ws.mlflow_tracking_uri)
    # mlflow.start_run()

    # enable autologging
    mlflow.sklearn.autolog()

    X = df.loc[:, df.columns != "label"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        # "n_estimators": 500,
        # "max_depth": 4,
        # "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }


    model = ensemble.HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    mse = mean_squared_error(y_test, model.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    mlflow.log_metric("MSE", mse)

    # Registering the model to the workspace
    # print("Registering the model via MLFlow")
    # mlflow.sklearn.log_model(
    #     sk_model=model,
    #     registered_model_name=args.registered_model_name,
    #     artifact_path=args.registered_model_name
    # )

    # # Saving the model to a file
    print("Saving the model via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=model,
        # path=os.path.join(args.registered_model_name, "trained_model"),
        path=args.model_output
    )
    ###########################
    #</save and register model>
    ###########################
    # mlflow.end_run()

if __name__ == '__main__':

    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"AML train data path: {args.data_path}"
    ]

    for line in lines:
        print(line)
    
    args = parse_args()

    main(args)

    mlflow.end_run()
