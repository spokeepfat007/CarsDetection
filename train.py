"""
Main file for training Yolo model

"""

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from model import Yolov1, YoloBody
from dataset import CustomDataset
from utils import (
    mean_average_precision,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,
    model_save
)
from loss import YoloLoss

torch.manual_seed(config.seed)

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"loss per epoch was {sum(mean_loss)/len(mean_loss)}")


def main():
    best_test_map = 0
    model = YoloBody().to(config.DEVICE)
    child = model.children().__next__()
    for param in child.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss(C=1)

    if config.LOAD_MODEL:
        load_checkpoint(torch.load(config.LOAD_MODEL_FILE), model, optimizer)

    train_dataset = CustomDataset(
        "train.csv",
        transform=config.train_transforms,
        img_dir=config.IMG_DIR,
    )

    test_dataset = CustomDataset(
        "test.csv", transform=config.test_transforms, img_dir=config.IMG_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(config.EPOCHS):

        train_fn(train_loader, model, optimizer, loss_fn)
        if not epoch%5:
            pred_boxes, target_boxes = get_bboxes(
                test_loader, model, iou_threshold=0.5, threshold=0.4
            )

            mean_avg_prec1 = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4
            )

            mean_avg_prec2 = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Train mAP: {mean_avg_prec1,mean_avg_prec2}")
            if best_test_map<mean_avg_prec1:
                save_checkpoint(model, "weights/best_yolov4.pth.tar")
                best_test_map = mean_avg_prec1
        if not epoch%10:
            save_checkpoint(config.LOAD_MODEL_FILE, model, optimizer)
        if epoch == config.FREEZE:
            child = model.children().__next__()
            for param in child.parameters():
                param.requires_grad = True



if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
