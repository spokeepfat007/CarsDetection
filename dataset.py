"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
import re


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
            self, csv_file, img_dir, S=13, B=2, C=1, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = self.annotations.iloc[index, 0]
        file = open(label_path)
        line = file.read()
        words = line.split()
        boxes = []
        for i in range(1, len(words), 4):
            params = [float(x) for x in words[i:i + 4]]
            boxes.append(params)
        img_path = "data/training_images/" + words[0]
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        for i in range(4):
            boxes[..., i] /= image.size[i % 2]
        image = np.array(image.convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=boxes)
            image = augmentations["image"]
            boxes = augmentations["bboxes"]

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            x, y, width, height = box
            class_label = 1

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )


            try:
                if label_matrix[i, j, 1] == 0:

                    label_matrix[i, j, 1] = 1

                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    label_matrix[i, j, 2:6] = box_coordinates


                    label_matrix[i, j, 1] = 1
            except Exception:
                print(i, j, x, y)

        return image, label_matrix


def makeDataset():
    annotations = pd.read_csv("data/train.csv")
    train = annotations[:int(len(annotations) * 0.7)]
    test = annotations[int(len(annotations) * 0.7):]
    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)


if __name__ == "__main__":
    makeDataset()
