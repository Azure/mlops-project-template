import argparse
from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from azureml.core import Run, Model

from fairlearn.metrics._group_metric_set import _create_group_metric_set
from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id

from interpret_community import TabularExplainer
from azureml.interpret import ExplanationClient

import mlflow
import mlflow.sklearn

# current run 
run = Run.get_context()
ws = run.experiment.workspace

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
        
    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_name", type=str, help="Name of registered model")
    parser.add_argument("--model_input", type=str, help="Path of input model")
    parser.add_argument("--prepared_data", type=str, help="Path to transformed data")
    parser.add_argument("--predictions", type=str, help="Path of predictions")
    parser.add_argument("--score_report", type=str, help="Path to score report")
    parser.add_argument('--deploy_flag', type=str, help='A deploy flag whether to deploy or no')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    lines = [
        f"Model path: {args.model_input}",
        f"Test data path: {args.prepared_data}",
        f"Predictions path: {args.predictions}",
        f"Scoring output path: {args.score_report}",
    ]

    for line in lines:
        print(line)

    # ---------------- Model Evaluation ---------------- #

    # Load the test data

    print("mounted_path files: ")
    arr = os.listdir(args.prepared_data)

    train_data = pd.read_csv((Path(args.prepared_data) / "train.csv"))
    test_data = pd.read_csv((Path(args.prepared_data) / "test.csv"))

    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    y_test = test_data[TARGET_COL]
    X_test = test_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # Load the model from input port
    model = pickle.load(open((Path(args.model_input) / "model.pkl"), "rb"))

    # Get predictions to y_test (y_test)
    yhat_test = model.predict(X_test)

    # Save the output data with feature columns, predicted cost, and actual cost in csv file
    output_data = X_test.copy()
    output_data["real_label"] = y_test
    output_data["predicted_label"] = yhat_test
    output_data.to_csv((Path(args.predictions) / "predictions.csv"))

    # Evaluate Model performance with the test set
    r2 = r2_score(y_test, yhat_test)
    mse = mean_squared_error(y_test, yhat_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, yhat_test)

    # Print score report to a text file
    (Path(args.score_report) / "score.txt").write_text(
        "Scored with the following model:\n{}".format(model)
    )
    with open((Path(args.score_report) / "score.txt"), "a") as f:
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
    plt.title("Comparing Model Predictions to Real values - Test Data")
    plt.savefig("predictions.png")
    mlflow.log_artifact("predictions.png")

    # -------------------- Promotion ------------------- #
    scores = {}
    predictions = {}
    score = r2_score(y_test, yhat_test) # current model
    for model_run in Model.list(ws):
        if model_run.name == args.model_name:
            model_path = Model.download(model_run, exist_ok=True)
            mdl = pickle.load(open((Path(model_path) / "model.pkl"), "rb"))
            predictions[model_run.id] = mdl.predict(X_test)
            scores[model_run.id] = r2_score(y_test, predictions[model_run.id])
        
    print(scores)
    if scores:
        if score >= max(list(scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print("Deploy flag: ",deploy_flag)

    with open((Path(args.deploy_flag) / "deploy_flag"), 'w') as f:
        f.write('%d' % int(deploy_flag))
                
    scores["current model"] = score
    perf_comparison_plot = pd.DataFrame(scores, index=["r2 score"]).plot(kind='bar', figsize=(15, 10))
    perf_comparison_plot.figure.savefig("perf_comparison.png")
    perf_comparison_plot.figure.savefig(Path(args.score_report) / "perf_comparison.png")
    
    mlflow.log_metric("deploy flag", bool(deploy_flag))
    mlflow.log_artifact("perf_comparison.png")
    

    # -------------------- FAIRNESS ------------------- #
    # Calculate Fairness Metrics over Sensitive Features
    # Create a dictionary of model(s) you want to assess for fairness 
    
    sf = { col: X_test[[col]] for col in SENSITIVE_COLS }
    predictions["currrent model"] = [x for x in model.predict(X_test)]
    
    dash_dict_all = _create_group_metric_set(y_true=y_test,
                                             predictions=predictions,
                                             sensitive_features=sf,
                                             prediction_type='regression',
                                            )
    
    # Upload the dashboard to Azure Machine Learning
    dashboard_title = "Fairness insights Comparison of Models"

    # Set validate_model_ids parameter of upload_dashboard_dictionary to False 
    # if you have not registered your model(s)
    upload_id = upload_dashboard_dictionary(run,
                                            dash_dict_all,
                                            dashboard_name=dashboard_title,
                                            validate_model_ids=False)
    print("\nUploaded to id: {0}\n".format(upload_id))

    
    # -------------------- Explainability ------------------- #
    tabular_explainer = TabularExplainer(model,
                                   initialization_examples=X_train,
                                   features=X_train.columns)
    
    # save explainer                                 
    #joblib.dump(tabular_explainer, os.path.join(tabular_explainer, "explainer"))

    # find global explanations for feature importance
    # you can use the training data or the test data here, 
    # but test data would allow you to use Explanation Exploration
    global_explanation = tabular_explainer.explain_global(X_test)

    # sorted feature importance values and feature names
    sorted_global_importance_values = global_explanation.get_ranked_global_values()
    sorted_global_importance_names = global_explanation.get_ranked_global_names()

    print("Explainability feature importance:")
    # alternatively, you can print out a dictionary that holds the top K feature names and values
    global_explanation.get_feature_importance_dict()
    
    client = ExplanationClient.from_run(run)
    client.upload_model_explanation(global_explanation, comment='global explanation: all features')


if __name__ == "__main__":
    main()