import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import config as cfg
from model import SOTModel
from utils import plot_examples
from data import Data, collate, reverse_transform


def evaluate():
    model = SOTModel()
    model = model.load_from_checkpoint(cfg.CHECKPOINT).evaluate()
    print("Model evaluated successfully.")

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

    org_imgs = []
    pred_imgs = []
    org_bboxes = []
    pred_bboxes = []

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


if __name__ == "__main__":
    evaluate()
