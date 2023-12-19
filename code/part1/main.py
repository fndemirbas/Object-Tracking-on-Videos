import torch
import config
import model
import test as t


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
learning_rate = 0.001


train_images, train_boxes, train_labels = model.load_data(config.train_dir)
dataset = model.Dataset(train_images, train_labels, train_boxes)

val_images, val_boxes, val_labels = model.load_data(config.val_dir)
valdataset =model.ValDataset(val_images, val_labels, val_boxes)



t.Test(config.test_dir)