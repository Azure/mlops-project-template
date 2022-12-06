# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import argparse

import pandas as pd

import mlflow
import torch
from sklearn import metrics as sklmetrics
import matplotlib.pyplot as plt

from model import CustomImageDataset, load_model


def main(labels_path, images_path, model_path, model_name, output_dir, deploy_flag_output):

    # Load model
    model_file = os.path.join(model_path, f'{model_name}.pth')
    net = load_model(path=model_file)

    # Load test data
    labels_data = pd.read_csv(os.path.join(labels_path, 'labels_test.csv'))
    labels_data = labels_data.set_index('path').squeeze() # Convert to appropiate format for CustomImageDataset
    testset = CustomImageDataset(images_path, labels_data, mode='test')
    print(f'Test size: {len(testset)}')

    # Generate predictions
    predictions = get_predictions(testset, net)
    predictions.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    # Evaluation metrics
    metrics = evaluate(predictions.label_real, predictions.label_predicted)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Plot confusion matrix
    plt.rcParams["figure.figsize"] = [40, 40]
    sklmetrics.ConfusionMatrixDisplay.from_predictions(predictions.label_real, predictions.label_predicted, cmap='YlGnBu')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Promote model
    deploy_flag = is_new_model_better()
    mlflow.log_metric("deploy flag", deploy_flag)
    with open(deploy_flag_output, 'w') as f:
        f.write('%d' % int(deploy_flag))

    deploy_flag_str = 'not' if deploy_flag == False else ''
    print(f'Finished. Model will {deploy_flag_str} be registered.')


def get_predictions(dataset, net, batch_size=256):

    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    predictions = pd.DataFrame()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, classes, paths = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            predictions_batch = pd.DataFrame({
                'path': paths,
                'label_real': dataset.get_labels(classes),
                'label_predicted': dataset.get_labels(predicted)
            })
            predictions = pd.concat([predictions, predictions_batch], ignore_index=True)

    return predictions


def evaluate(labels_real, labels_pred):

    metrics = {
        'accuracy': sklmetrics.accuracy_score(labels_real, labels_pred),
        'mcc': sklmetrics.matthews_corrcoef(labels_real, labels_pred),

        'recall_micro': sklmetrics.recall_score(labels_real, labels_pred, average='micro'),
        'recall_macro': sklmetrics.recall_score(labels_real, labels_pred, average='macro'),
        'recall_weighted': sklmetrics.recall_score(labels_real, labels_pred, average='weighted'),

        'precison_micro': sklmetrics.precision_score(labels_real, labels_pred, average='micro'),
        'precison_macro': sklmetrics.precision_score(labels_real, labels_pred, average='macro'),
        'precison_weighted': sklmetrics.precision_score(labels_real, labels_pred, average='weighted'),

        'f1_micro': sklmetrics.f1_score(labels_real, labels_pred, average='micro'),
        'f1_macro': sklmetrics.f1_score(labels_real, labels_pred, average='macro'),
        'f1_weighted': sklmetrics.f1_score(labels_real, labels_pred, average='weighted'),
    }

    return metrics


def is_new_model_better():
    return True  # For simplicity


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared_data_path', type=str, required=True, help='Directory path to test data (output from prep step)')
    parser.add_argument('--dogs-imgs', type=str, help='Directory path to images')
    parser.add_argument('--model_path', type=str, help='Model output directory')
    parser.add_argument('--evaluation_path', type=str, default='evaluation_results/', help="Evaluation results output directory")
    parser.add_argument('--deploy_flag', type=str, help='A deploy flag whether to deploy or no')

    args_parsed, unknown = parser.parse_known_args(args_list)
    if unknown:
        print(f"Unrecognized arguments. These won't be used: {unknown}")

    return args_parsed


if __name__ == '__main__':
    args = parse_args()

    main(
        labels_path=args.prepared_data_path,
        images_path=args.dogs_imgs,
        model_path=args.model_path,
        model_name='model',
        output_dir=args.evaluation_path,
        deploy_flag_output=args.deploy_flag
    )
