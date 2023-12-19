import sys
import copy
import collections
from pathlib import Path
from typing import Tuple
from natsort import natsorted

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, dataloader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import config as cfg


Rectangle = collections.namedtuple("Rectangle", ["x", "y", "width", "height"])
Point = collections.namedtuple("Point", ["x", "y"])
Polygon = collections.namedtuple("Polygon", ["points"])


def convert_to_region(region, to):

    if to == "rectangle":

        if isinstance(region, Rectangle):
            return copy.copy(region)
        elif isinstance(region, Polygon):
            top = sys.float_info.max
            bottom = sys.float_info.min
            left = sys.float_info.max
            right = sys.float_info.min

            for point in region.points:
                top = min(top, point.y)
                bottom = max(bottom, point.y)
                left = min(left, point.x)
                right = max(right, point.x)

            return Rectangle(left, top, right - left, bottom - top)

        else:
            return None
    if to == "polygon":

        if isinstance(region, Rectangle):
            points = []
            points.append((region.x, region.y))
            points.append((region.x + region.width, region.y))
            points.append((region.x + region.width, region.y + region.height))
            points.append((region.x, region.y + region.height))
            return Polygon(points)

        elif isinstance(region, Polygon):
            return copy.copy(region)
        else:
            return None

    return None


def convert_to_box(bbox):
    if len(bbox) == 4:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        return [x1, y1, x2, y2]

    elif len(bbox) > 4:
        pts = []
        for idx in range(0, len(bbox), 2):
            pts.append(Point(bbox[idx], bbox[idx + 1]))
        poly = Polygon(pts)
        rect = convert_to_region(poly, "rectangle")
        x1, y1, w, h = rect.x, rect.y, rect.width, rect.height
        x2, y2 = x1 + w, y1 + h
        return [x1, y1, x2, y2]


class Data(Dataset):
    def __init__(self, data_dir: Path = Path(cfg.DATA_DIR)):

        objects = [
            obj
            for obj in list(data_dir.glob("*"))
            if obj.is_dir() and not str(obj.name).startswith(".")
        ]

        data = []

        for obj in objects:
            img_path = obj / "color"
            annot_path = obj / "groundtruth.txt"

            images = natsorted(list(img_path.glob("*")))

            with open(str(annot_path), "r") as fl:
                annots = fl.read()
                annots = annots.split("\n")
                annots = [
                    [float(coord) for coord in annot.split(",")]
                    for annot in annots
                    if annot != ""
                ]

            annots = list(map(convert_to_box, annots))

            data += list(
                zip(
                    images[:-1],
                    annots[:-1],
                    images[1:],
                    annots[1:],
                    [obj.name] * (len(images) - 1),
                )
            )

        self.data = data
        self.transform_x = A.Compose(
            [
                A.RandomCropNearBBox(always_apply=True),
                A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
                ToTensorV2(),
            ],
            p=1.0,
        )

        self.transform_y = A.Compose(
            [
                A.RandomSizedBBoxSafeCrop(cfg.IMG_SIZE, cfg.IMG_SIZE),
                ToTensorV2(),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=[], min_visibility=0.3
            ),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_x, bbox_x, image_y, bbox_y, obj = self.data[index]

        image_x = np.array(Image.open(image_x))
        bbox_x = np.array(bbox_x, dtype=np.float32)

        image_y = np.array(Image.open(image_y))
        bbox_y = np.array(bbox_y, dtype=np.float32)

        try:
            if self.transform_x:
                transformed = self.transform_x(image=image_x, cropping_bbox=bbox_x)
                image_x = transformed["image"]

            if self.transform_y:
                transformed = self.transform_y(image=image_y, bboxes=[bbox_y])
                image_y = transformed["image"]
                bbox_y = torch.tensor(transformed["bboxes"][0]).float()

            return {
                "previous_frame": image_x.float(),  # previous frame
                "current_frame": image_y.float(),  # current frame
                "bbox": bbox_y,  # target
                "name": obj,  # object name
            }
        except:
            return None


def collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return dataloader.default_collate(batch)


def reverse_transform(
    img: torch.Tensor,
    bbox: torch.Tensor,
    width: int = None,
    height: int = None,
) -> Tuple[np.ndarray, np.ndarray]:

    if width == None:
        width = img.shape[1]
    if height == None:
        height = img.shape[2]

    img = img.permute(1, 2, 0).numpy()
    bbox = bbox.numpy()

    transform = A.Compose(
        [
            A.Resize(width, height),
        ],
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=[], min_visibility=0.4),
    )

    transformed = transform(image=img, bboxes=[bbox])

    return transformed["image"], transformed["bboxes"][0]
