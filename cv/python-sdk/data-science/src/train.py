# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Example adapted from:
#   https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#   https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import os
import argparse

import pandas as pd
import mlflow
import torch

from model import CustomImageDataset, load_model


def main(labels_path, images_path, 
         model_name, output_dir,
         mode, epochs, batch_size, learning_rate, momentum):

    labels_data = pd.read_csv(os.path.join(labels_path, 'labels_train.csv'))
    labels_data = labels_data.set_index('path').squeeze() # Convert to appropiate format for CustomImageDataset
    trainset = CustomImageDataset(images_path, labels_data, mode='train')
    print(f'Train size: {len(trainset)}')

    print("Training...")
    net = train(
        trainset,
        mode=mode,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum
    )
    print('Finished training')

    print(f"Saving model in folder {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_name}.pth')
    torch.save(net.state_dict(), model_path)

    print('Finished.')


def train(dataset, mode='finetuning', epochs=2, batch_size=64, learning_rate=0.001, momentum=0.9, stats_freq=25):

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    net, criterion, optimizer = load_model(
        num_classes=dataset.nclasses(),
        mode=mode,
        learning_rate=learning_rate,
        momentum=momentum
    )

    global_iter = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f'----\nEpoch {epoch}\n----\n')

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, classes]
            inputs, classes, paths = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, classes)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % stats_freq == 0:
                mlflow.log_metric('loss', running_loss / stats_freq, step=global_iter)
                mlflow.log_metric('epoch', epoch, step=global_iter)
                running_loss = 0.0

            global_iter += 1

    return net


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared_data_path', type=str, required=True, help='Directory path to training data (output from prep step)')
    parser.add_argument('--dogs-imgs', type=str, help='Directory path to images')
    parser.add_argument('--model_path', type=str, help='Model output directory')
    parser.add_argument('--training-mode', type=str, default='feature-extraction', choices=['finetuning', 'feature-extraction'])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)

    args_parsed, unknown = parser.parse_known_args(args_list)
    if unknown:
        print(f"Unrecognized arguments. These won't be used: {unknown}")

    return args_parsed


if __name__ == '__main__':
    args = parse_args()

    main(
        labels_path=args.prepared_data_path,
        images_path=args.dogs_imgs,
        model_name='model',
        output_dir=args.model_path,
        mode=args.training_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum
    )
