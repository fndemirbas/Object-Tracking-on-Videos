from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import csv

from matplotlib import patches

import train

def Test(test_dir):
    x_min, y_min, x_max, y_max = 0, 0,0,0
    with open(test_dir + '/_annotations.txt', 'r') as file:
        my_reader = csv.reader(file, delimiter=' ')
        for row in my_reader:
            image_dir = test_dir + '/' + row[0]
            image = Image.open(image_dir)
            x, y = image.size

            image = image.resize((64, 64))
            #image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)
            x1, y1 = image.size
            rx, ry = x1 / x, y1 / y

            if len(row[1].split(',')) == 5:
                coords = row[1].split(',')
                box = []
                image = np.array(image)

                x_min = int(float(coords[0]) * rx)
                y_min = int(float(coords[1]) * ry)
                x_max = int(float(coords[2]) * rx)
                y_max = int(float(coords[3]) * ry)

            x1, y1, x2, y2 =  train.predict(image_dir)

            rect = patches.Rectangle((x1, y1), x2, y2, linewidth=1, edgecolor='g', facecolor='none')

            fig, ax = plt.subplots(1)
            ax.add_patch(rect)

            rect = patches.Rectangle((x_min, y_min), x_max, y_max, linewidth=1,  edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.imshow(image)
            plt.show()
