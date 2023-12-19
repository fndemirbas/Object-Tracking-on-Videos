import os
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

import config as cfg
from model import SOTModel
from data import Data, collate


def main():
    ds = Data()
    val_size = int(len(ds) * cfg.VAL_SIZE)
    test_size = int(len(ds) * cfg.TEST_SIZE)
    train_size = len(ds) - val_size - test_size

    train_ds, val_ds, test_ds = random_split(ds, [train_size, val_size, test_size])

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        collate_fn=collate,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        num_workers=os.cpu_count(),
        collate_fn=collate,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        num_workers=os.cpu_count(),
        collate_fn=collate,
    )

    model = SOTModel()
    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=cfg.EPOCHS,
        precision='bf16',
        logger=pl.loggers.TensorBoardLogger("logs/", name="sot", version=cfg.VERSION),
    )

    trainer.fit(
        model,
        train_dl,
        val_dl,
    )
    trainer.test(test_dataloaders=test_dl)

    trainer.save_checkpoint(f"checkpoints/sot_{cfg.VERSION}.ckpt")


if __name__ == "__main__":
    main()
