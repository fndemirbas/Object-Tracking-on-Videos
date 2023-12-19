from PIL import Image
import torch
import torchvision
import numpy as np

import torch.nn as nn
import csv

class Dataset():
    def __init__(self, train_images, train_labels, train_boxes):
        self.images = torch.permute(torch.from_numpy(train_images),(0,3,1,2)).float()
        self.labels = torch.from_numpy(train_labels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(train_boxes).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.images[idx],
              self.labels[idx],
              self.boxes[idx])


class ValDataset(Dataset):
    def __init__(self, val_images, val_labels, val_boxes):
        self.images = torch.permute(torch.from_numpy(val_images),(0,3,1,2)).float()
        self.labels = torch.from_numpy(val_labels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(val_boxes).float()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        self.fc1 = nn.Sequential(*self.getRequireFeatures(model))
        self.fc_classifier = nn.Sequential(nn.Linear(64 * 16 * 16, 2), nn.ReLU())
        self.box = nn.Sequential(nn.Linear(64 * 16 * 16, 4), nn.ReLU())


    def forward(self, X):
        X = self.fc1(X)
        X = X.reshape(-1, 64 * 16 * 16)
        predClass = self.fc_classifier(X)
        box = self.box(X)
        return predClass, box


    def getRequireFeatures(self, model):
        fc = list(model.children())
        req_features = []
        k = torch.zeros([1, 3, 64, 64]).float()
        for i in fc:
            k = i(k)
            if k.size()[2] < 800 // 80:
                break
            req_features.append(i)
        return req_features


def get_num_correct(preds, labels):
    return torch.round(preds).argmax(dim=1).eq(labels).sum().item()


def load_data(dir):
    temp =[]
    boxes = []
    labels = []

    with open(dir + '/_annotations.txt', 'r') as file:
        my_reader = csv.reader(file, delimiter=' ')
        for row in my_reader:
            image= Image.open( dir + '/' + row[0])
            x, y = image.size

            image = image.resize((64, 64))
            x1, y1 = image.size
            rx, ry = x1 / x, y1 / y

            if len(row[1].split(',')) == 5:
                coords = row[1].split(',')
                box = []
                image = np.array(image)
                temp.append(image)

                box.append(float(coords[0]) * rx)
                box.append(float(coords[1]) * ry)
                box.append(float(coords[2]) * rx)
                box.append(float(coords[3]) * ry)
                boxes.append(box)
                labels.append(0)

    return np.array(temp), np.array(boxes), np.array(labels)