"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):

    def __init__(self, S=13, B=2, C=1):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S - 
        B - количество боксов на каждую сетку
        C  - число классов
        """
        self.S = S
        self.B = B
        self.C = C

        # Параметры из статьи
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Считаем IoU индекс для B боксов и targetа
        ious = [intersection_over_union(predictions[..., self.C + 1 + i * 5:self.C + 5 * (i + 1)],
                                        target[..., self.C + 1:self.C + 5]).unsqueeze(0) for i in range(self.B)]
        ious = torch.cat(ious, dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., self.C].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = torch.zeros_like(predictions[..., self.C + 1:self.C + 5])
        for i in range(self.B):
            box_predictions += (bestbox == i) * predictions[..., self.C + 1 + 5 * i:self.C + 5 * (i + 1)]
        box_predictions = exists_box * box_predictions

        box_targets = exists_box * target[..., self.C + 1:self.C + 5]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = torch.zeros_like(predictions[..., self.C:self.C + 1])
        for i in range(self.B):
            pred_box += (
                    (bestbox == i) * predictions[..., self.C + 5 * i:self.C + 1 + 5 * i]
            )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:self.C + 1]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = 0
        for i in range(self.B):
            no_object_loss += self.mse(
                torch.flatten((1 - exists_box) * predictions[..., self.C+5*i:self.C+1+5*i], start_dim=1),
                torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1),
            )
        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2, ),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2, ),
        )

        loss = (
                self.lambda_coord * box_loss  # first two rows in paper
                + object_loss  # third row in paper
                + self.lambda_noobj * no_object_loss  # forth row
                + class_loss  # fifth row
        )

        return loss
