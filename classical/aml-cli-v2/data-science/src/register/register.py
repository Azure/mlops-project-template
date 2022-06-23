# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers trained ML model if deploy flag is True.
"""

import argparse
from pathlib import Path
import pickle
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument('--evaluation_output', type=str, help='Path of eval results')

    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args


def main():
    '''Loads model, registers it if deply flag is True'''

    mlflow.start_run()

    args = parse_args()

    model_name = args.model_name
    model_path = args.model_path

    with open((Path(args.evaluation_output) / "deploy_flag"), 'rb') as infile:
        deploy_flag = int(infile.read())
    
    mlflow.log_metric("deploy flag", bool(deploy_flag))
    deploy_flag=1 
    if deploy_flag==1:

        print("Registering ", model_name)

        # load model
        with open((Path(model_path) / "model.pkl"), "rb") as infile:
            model = pickle.load(infile)

        # log model using mlflow
        mlflow.sklearn.log_model(model, model_name)

        # register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f'runs:/{run_id}/{args.model_name}'
        mlflow.register_model(model_uri, model_name)

    else:
        print("Model will not be registered!")

    mlflow.end_run()

if __name__ == "__main__":
    main()
