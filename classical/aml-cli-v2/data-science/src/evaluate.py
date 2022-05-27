import argparse
from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from azureml.core import Run, Model

import mlflow
import mlflow.sklearn

# current run 
run = Run.get_context()
ws = run.experiment.workspace

def parse_args():
        
    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_name", type=str, help="Name of registered model")
    parser.add_argument("--model_input", type=str, help="Path of input model")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--predictions", type=str, help="Path of predictions")
    parser.add_argument("--score_report", type=str, help="Path to score report")
    parser.add_argument('--deploy_flag', type=str, help='A deploy flag whether to deploy or no')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    lines = [
        f"Model path: {args.model_input}",
        f"Test data path: {args.test_data}",
        f"Predictions path: {args.predictions}",
        f"Scoring output path: {args.score_report}",
    ]

    for line in lines:
        print(line)

    # ---------------- Model Evaluation ---------------- #

    # Load the test data

    print("mounted_path files: ")
    arr = os.listdir(args.test_data)

    print(arr)

    test_data = pd.read_csv((Path(args.test_data) / "test.csv"))
    print(test_data.columns)

    y_test = test_data["cost"]
    # X_test = test_data.drop(['cost'], axis=1)
    X_test = test_data[
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
    print(X_test.shape)
    print(X_test.columns)

    # Load the model from input port
    model = pickle.load(open((Path(args.model_input) / "model.pkl"), "rb"))
    # model = (Path(args.model_input) / 'model.txt').read_text()
    # print('Model: ', model)

    # Compare predictions to y_test (y_test)
    output_data = X_test.copy()
    output_data["actual_cost"] = y_test
    output_data["predicted_cost"] = model.predict(X_test)

    # Save the output data with feature columns, predicted cost, and actual cost in csv file
    output_data.to_csv((Path(args.predictions) / "predictions.csv"))

    # Print the results of scoring the predictions against actual values in the test data
    # The coefficients
    print("Coefficients: \n", model.coef_)

    y_test = output_data["actual_cost"]
    yhat_test = output_data["predicted_cost"]

    # Evaluate Linear Regression performance with the test set
    r2 = r2_score(y_test, yhat_test)
    mse = mean_squared_error(y_test, yhat_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, yhat_test)

    # Print score report to a text file
    (Path(args.score_report) / "score.txt").write_text(
        "Scored with the following model:\n{}".format(model)
    )
    with open((Path(args.score_report) / "score.txt"), "a") as f:
        f.write("\n Coefficients: \n %s \n" % str(model.coef_))
        f.write("Mean squared error: %.2f \n" % mse)
        f.write("Root mean squared error: %.2f \n" % rmse)
        f.write("Mean absolute error: %.2f \n" % mae)
        f.write("Coefficient of determination: %.2f \n" % r2)


    mlflow.log_metric("test r2", r2)
    mlflow.log_metric("test mse", mse)
    mlflow.log_metric("test rmse", rmse)
    mlflow.log_metric("test mae", mae)

    # Visualize results
    plt.scatter(y_test, yhat_test,  color='black')
    plt.plot(y_test, y_test, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # -------------------- Promotion ------------------- #
    test_scores = {}
    test_predictions = {}
    test_score = r2_score(y_test, yhat_test) # current model
    for model_run in Model.list(ws):
        if model_run.name == args.model_name:
            model_path = Model.download(model_run, exist_ok=True)
            mdl = pickle.load(open((Path(model_path) / "model.pkl"), "rb"))
            test_predictions[model_run.id] = mdl.predict(X_test)
            test_scores[model_run.id] = r2_score(y_test, test_predictions[model_run.id])
        
    print(test_scores)
    if test_scores:
        if test_score >= max(list(test_scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print("Deploy flag: ",deploy_flag)

    with open((Path(args.deploy_flag) / "deploy_flag"), 'w') as f:
        f.write('%d' % int(deploy_flag))
                
    test_scores["current model"] = test_score
    model_runs_metrics_plot = pd.DataFrame(test_scores, index=["r2 score"]).plot(kind='bar', figsize=(15, 10))
    model_runs_metrics_plot.figure.savefig("model_runs_metrics_plot.png")
    model_runs_metrics_plot.figure.savefig(Path(args.score_report) / "model_runs_metrics_plot.png")
    
    mlflow.log_metric("deploy flag", bool(deploy_flag))
    mlflow.log_artifact("model_runs_metrics_plot.png")
    

if __name__ == "__main__":
    main()