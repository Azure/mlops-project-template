# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Evaluates trained ML model using test dataset.
Saves predictions, evaluation results and deploy flag.
"""

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from fairlearn.metrics._group_metric_set import _create_group_metric_set
from interpret_community import TabularExplainer

from azureml.core import Run
from azureml.contrib.fairness import upload_dashboard_dictionary
from azureml.interpret import ExplanationClient

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
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

CAT_NOM_COLS = [
    "store_forward",
    "vendor",
]

CAT_ORD_COLS = [
]

SENSITIVE_COLS = ["vendor"] # for fairlearn dashborad


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_name", type=str, help="Name of registered model")
    parser.add_argument("--model_input", type=str, help="Path of input model")
    parser.add_argument("--prepared_data", type=str, help="Path to transformed data")
    parser.add_argument("--evaluation_output", type=str, help="Path of eval results")

    args = parser.parse_args()

    return args


def main(prepared_data, model_input, evaluation_output):
    '''Read trained model and test dataset, evaluate model and save result'''

    # ---------------- Model Evaluation ---------------- #

    # Load the train and test data
    train_data = pd.read_csv((Path(prepared_data) / "train.csv"))
    test_data = pd.read_csv((Path(prepared_data) / "test.csv"))

    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    y_test = test_data[TARGET_COL]
    X_test = test_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # Load the model from input port
    with open((Path(model_input) / "model.pkl"), "rb") as infile:
        model = pickle.load(infile)

    # Get predictions to y_test (y_test)
    yhat_test = model.predict(X_test)

    # Save the output data with feature columns, predicted cost, and actual cost in csv file
    output_data = X_test.copy()
    output_data["real_label"] = y_test
    output_data["predicted_label"] = yhat_test
    output_data.to_csv((Path(evaluation_output) / "predictions.csv"))

    # Evaluate Model performance with the test set
    r2 = r2_score(y_test, yhat_test)
    mse = mean_squared_error(y_test, yhat_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, yhat_test)

    # Print score report to a text file
    (Path(evaluation_output) / "score.txt").write_text(
        f"Scored with the following model:\n{format(model)}"
    )
    with open((Path(evaluation_output) / "score.txt"), "a") as outfile:
        outfile.write("Mean squared error: {mse.2f} \n")
        outfile.write("Root mean squared error: {rmse.2f} \n")
        outfile.write("Mean absolute error: {mae.2f} \n")
        outfile.write("Coefficient of determination: {r2.2f} \n")

    mlflow.log_metric("test r2", r2)
    mlflow.log_metric("test mse", mse)
    mlflow.log_metric("test rmse", rmse)
    mlflow.log_metric("test mae", mae)

    # Visualize results
    plt.scatter(y_test, yhat_test,  color='black')
    plt.plot(y_test, y_test, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.title("Comparing Model Predictions to Real values - Test Data")
    plt.savefig("predictions.png")
    mlflow.log_artifact("predictions.png")

    # -------------------- Promotion ------------------- #
    scores = {}
    predictions = {}
    score = r2_score(y_test, yhat_test) # current model

    client = MlflowClient()

    for model_run in client.search_model_versions("name='taxi-model'"):
        model_name = model_run.name
        model_version = model_run.version
        mdl = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}")
        predictions[f"{model_name}:{model_version}"] = mdl.predict(X_test)
        scores[f"{model_name}:{model_version}"] = r2_score(
            y_test, predictions[f"{model_name}:{model_version}"])

    if scores:
        if score >= max(list(scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print(f"Deploy flag: {deploy_flag}")

    with open((Path(evaluation_output) / "deploy_flag"), 'w') as outfile:
        outfile.write(f"{int(deploy_flag)}")

    # add current model score and predictions
    scores["current model"] = score
    predictions["currrent model"] = model.predict(X_test)

    perf_comparison_plot = pd.DataFrame(
        scores, index=["r2 score"]).plot(kind='bar', figsize=(15, 10))
    perf_comparison_plot.figure.savefig("perf_comparison.png")
    perf_comparison_plot.figure.savefig(Path(evaluation_output) / "perf_comparison.png")

    mlflow.log_metric("deploy flag", bool(deploy_flag))
    mlflow.log_artifact("perf_comparison.png")

    # -------------------- FAIRNESS ------------------- #
    # add model fairness
    sensitive_features = { col: X_test[[col]] for col in SENSITIVE_COLS }
    fairness(sensitive_features, y_test, predictions)

    # -------------------- Explainability ------------------- #
    # add model explainability
    #explainability(model, X_train, X_test)


def fairness(sensitive_features, y_test, predictions):
    '''Generates fairness metrics, uploads results to AML run fairness dashboard'''

    # Calculate Fairness Metrics over Sensitive Features
    # Create a dictionary of model(s) you want to assess for fairness

    dash_dict_all = _create_group_metric_set(y_true=y_test,
                                             predictions=predictions,
                                             sensitive_features=sensitive_features,
                                             prediction_type='regression',
                                            )

    # Upload the dashboard to Azure Machine Learning
    dashboard_title = "Fairness insights Comparison of Models"

    # Set validate_model_ids parameter of upload_dashboard_dictionary to False
    # if you have not registered your model(s)
    run = Run.get_context()
    upload_id = upload_dashboard_dictionary(run,
                                            dash_dict_all,
                                            dashboard_name=dashboard_title,
                                            validate_model_ids=False)
    print(f"\nUploaded to id: {format(upload_id)}\n")

def explainability(model, X_train, X_test):
    '''Generates explainer, uploads results to AML run explainability dashboard'''

    tabular_explainer = TabularExplainer(model,
                                    initialization_examples=X_train,
                                    features=X_train.columns,
                                    model_task="regression")

    # find global explanations for feature importance
    # you can use the training data or the test data here,
    # but test data would allow you to use Explanation Exploration
    global_explanation = tabular_explainer.explain_global(X_test)

    # upload results to AML run
    run = Run.get_context()
    client = ExplanationClient.from_run(run)
    client.upload_model_explanation(global_explanation, comment='global explanation: all features')


if __name__ == "__main__":

    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Model path: {args.model_input}",
        f"Test data path: {args.prepared_data}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)
    
    main(args.prepared_data, args.model_input, args.evaluation_output)

    mlflow.end_run()
