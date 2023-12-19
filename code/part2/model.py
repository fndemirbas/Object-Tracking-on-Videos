import torch
from torch import nn
from torchvision import models
from pytorch_lightning import LightningModule

import config as cfg


class SOTModel(LightningModule):
    def __init__(self, lr=cfg.LEARNING_RATE):
        super(SOTModel, self).__init__()

        self.x_cnn = nn.Sequential(
            *(list(models.resnet34(pretrained=True).children())[:-1])
        )
        self.y_cnn = nn.Sequential(
            *(list(models.resnet34(pretrained=True).children())[:-1])
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4),
        )

        self.lr = lr
        self.loss = nn.MSELoss()

    def forward(self, previous_frame, current_frame):
        x_feature = self.x_cnn(previous_frame)
        y_feature = self.y_cnn(current_frame)

        x_feature = self.flatten(x_feature)
        y_feature = self.flatten(y_feature)

        features = torch.cat([x_feature, y_feature], dim=1)

        return self.sigmoid_scale(self.fc(features), 0, cfg.IMG_SIZE)

    def training(self, batch, batch_idx):
        target = batch["bbox"]

        out = self(batch["previous_frame"], batch["current_frame"])
        return self.loss(out, target)

    def validation(self, batch, batch_idx):
        target = batch["bbox"]

        out = self(batch["previous_frame"], batch["current_frame"])
        self.log("val_mse", self.loss(out, target), on_step=True, on_epoch=True)

    def test(self, batch, batch_idx):
        target = batch["bbox"]

        out = self(batch["previous_frame"], batch["current_frame"])
        self.log("test_mse", self.loss(out, target), on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    @staticmethod
    def sigmoid_scale(x, lo, hi):
        return torch.sigmoid(x) * (hi - lo) + lo
