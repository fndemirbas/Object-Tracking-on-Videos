# Object-Tracking-on-Videos
I implemented and practice with object recognition. This assignment has two parts. In Part 1, I implemented the basic object recognition+localization pipeline. In Part 2, I used YOLOv3 object detection with mean-shift algorithm for tracking.

### These must be:

from PIL import Image
import torch
import torchvision
from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import cv2
import csv

----------------------------------------------------------------------------------------------------------------------------------------------------------
The dataset folder must be in the same directory as the python files or the path can be changed from the config.py file.
At the same time, the variables can be changed from the config.py file.
Part1:
Train, aval and test files can be run internally.
Part2:
It works with the main file.
