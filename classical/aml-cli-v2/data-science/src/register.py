# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse
from pathlib import Path
import pickle 

import mlflow

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument('--deploy_flag', type=str, help='A deploy flag whether to deploy or no')
    
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args


def main():

    args = parse_args()

    model_name = args.model_name
    model_path = args.model_path

    #model = pickle.load(open((Path(args.model_path) / "model.sav"), "rb"))
    #pickle.dump(model, open("model.sav", "wb"))

    with open((Path(args.deploy_flag) / "deploy_flag"), 'r') as f:
        deploy_flag = int(f.read())

    if deploy_flag==1:
        print("Registering ", model_name)

        model = pickle.load(open((Path(args.model_path) / "model.sav"), "rb"))
        mlflow.sklearn.log_model(model, args.model_name)
        model_uri = f'{args.model_name}'
        mlflow.register_model(model_uri,args.model_name)

        registered_model = mlflow.register_model(args.model_path, args.model_name)
    else:
        print("Model will not be registered!")

if __name__ == "__main__":
    main()
