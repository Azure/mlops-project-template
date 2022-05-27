
import argparse

from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn


def parse_args():

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--training_data", type=str, help="Path to training data")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    args = parser.parse_args()

    return args

def main():

    args = parse_args()
    
    lines = [
        f"Training data path: {args.training_data}",
        f"Model output path: {args.model_output}",
    ]

    for line in lines:
        print(line)

    print("mounted_path files: ")
    arr = os.listdir(args.training_data)
    print(arr)

    train_data = pd.read_csv((Path(args.training_data) / "train.csv"))
    print(train_data.columns)

    # Split the data into input(X) and output(y)
    y_train = train_data["cost"]
    # X = train_data.drop(['cost'], axis=1)
    X_train = train_data[
        [
            "distance",
            "dropoff_latitude",
            "dropoff_longitude",
            "passengers",
            "pickup_latitude",
            "pickup_longitude",
            "store_forward",
            "vendor",
            "pickup_weekday",
            "pickup_month",
            "pickup_monthday",
            "pickup_hour",
            "pickup_minute",
            "pickup_second",
            "dropoff_weekday",
            "dropoff_month",
            "dropoff_monthday",
            "dropoff_hour",
            "dropoff_minute",
            "dropoff_second",
        ]
    ]


    # Train a Linear Regression Model with the train set
    model = LinearRegression().fit(X_train, y_train)

    # Predict using the Linear Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Linear Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)

    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # Save the model
    pickle.dump(model, open((Path(args.model_output) / "model.pkl"), "wb"))

if __name__ == "__main__":
    main()
    