# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import PIL

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_labels, mode='test'):      
        self.img_dir = img_dir
        self.img_labels = img_labels
        self.classes = img_labels.unique().tolist()

        self.mode = mode
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
 
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_path = self.img_labels.index[idx]
        image = PIL.Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        image = self.transform(image)

        img_label = self.img_labels[idx]
        img_class = self.classes.index(img_label)

        return image, img_class, img_path

    def nclasses(self):
        return len(self.classes)

    def get_labels(self, indexes):
        return [self.classes[i] for i in indexes]
