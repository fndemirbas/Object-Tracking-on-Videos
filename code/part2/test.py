import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import config as cfg
from model import SOTModel
from utils import plot_examples
from data import Data, collate, reverse_transform


def test():
    data = Data()
    data_iter = iter(data)

    image_list = []
    bbox_list = []

    for i in range(5):
        sample = next(data_iter)
        img = sample["current_frame"].permute(1, 2, 0).numpy()
        bbox = sample["bbox"].numpy()

        img_org, bbox_org = reverse_transform(
            sample["current_frame"],
            sample["bbox"],
            480,
            720,
        )

        image_list.append(img)
        image_list.append(img_org)
        bbox_list.append(bbox)
        bbox_list.append(bbox_org)

    plot_examples(
        image_list,
        bbox_list,
    )


def test_model():
    model = SOTModel()
    model.load_from_checkpoint(cfg.CHECKPOINT)
    sample_x = torch.randn((4, 3, 224, 224))
    sample_y = torch.randn((4, 3, 224, 224))
    sample_bbox = torch.randn((4, 4))

    loss = nn.MSELoss()
    sample_out = model(sample_x, sample_y)

    print(sample_out)
    print(sample_out.shape, sample_bbox.shape)
    print(loss(sample_bbox, sample_out))


def test_main():

    ds = Data()
    val_sz = int(len(ds) * cfg.VAL_SIZE)
    train_sz = len(ds) - val_sz

    train_ds, val_ds = random_split(ds, [train_sz, val_sz])

    print(len(train_ds), len(val_ds))

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        num_workers=4,
        collate_fn=collate,
    )

    # checking input batches: passing
    for idx, batch in enumerate(train_dl):
        print(idx, batch.keys())

    # checking loss function: passing
    x = torch.randn((32, 4)).requires_grad_()
    y = torch.randn((32, 4)).requires_grad_()

    loss = nn.MSELoss()

    print(loss(x, y))


def test_prediction():
    model = SOTModel()
    model = model.load_from_checkpoint(cfg.CHECKPOINT).eval()
    print("Model tested successfully")

    ds = Data()
    split_idx = int(len(ds) * cfg.VAL_SIZE)
    indices = list(range(len(ds)))

    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    train_sampler, val_sampler = SubsetRandomSampler(
        train_indices
    ), SubsetRandomSampler(val_indices)

    val_dl = DataLoader(
        ds,
        batch_size=8,
        sampler=val_sampler,
        num_workers=4,
        collate_fn=collate,
        shuffle=False,
    )

    val_batch = next(iter(val_dl))

    with torch.no_grad():
        out = model(val_batch["previous_frame"], val_batch["current_frame"])
    imgs = []
    bboxes = []

    for idx in range(len(out)):
        org_img, org_bbox = reverse_transform(
            val_batch["current_frame"][idx],
            val_batch["bbox"][idx],
            480,
            720,
        )

        pred_img, pred_bbox = reverse_transform(
            val_batch["current_frame"][idx],
            out[idx],
            480,
            720,
        )

        imgs += [org_img, pred_img]
        bboxes += [org_bbox, pred_bbox]

    plot_examples(imgs, bboxes)


