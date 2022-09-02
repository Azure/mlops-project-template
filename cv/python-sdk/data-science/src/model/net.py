# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


def load_model(path=None, num_classes=2, mode='finetuning', learning_rate=0.001, momentum=0.9):

    # Load existing model
    if path:
        print('Loading existing model from path...')
        model_data = torch.load(path)
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, model_data['fc.weight'].shape[0])
        model.load_state_dict(model_data)
        return model

    # Initialize new model
    assert mode in ['finetuning', 'feature-extraction']

    model = models.resnet18(pretrained=True)
    if mode == 'feature-extraction':  # Freeze layers
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss()

    params_optim = model.parameters() if mode == 'finetuning' else model.fc.parameters() if mode == 'feature-extraction' else None
    optimizer = optim.SGD(params_optim, lr=learning_rate, momentum=momentum)

    return model, criterion, optimizer
