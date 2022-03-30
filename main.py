import cv2
import numpy as np
import torch

import config
from utils import cellboxes_to_boxes, non_max_suppression

test_transforms = config.transforms


def main():
    img_path = "data/training_images/vid_4_12140.jpg"
    image = cv2.imread(img_path)
    image_arr = np.array(image)
    transform = test_transforms(image=image_arr)
    x = (transform["image"]).unsqueeze(0).to(config.DEVICE)
    # model = YoloBody().to(DEVICE)
    model = torch.load("weights/best_yolov4(2).pth.tar")
    model.eval()
    #now = datetime.now()

    with torch.no_grad():
        preds = model(x)

    boxes = cellboxes_to_boxes(preds)

    nms_boxes = non_max_suppression(
        boxes[0],
        iou_threshold=0.5,
        threshold=0.4,
        box_format="midpoint",
    )
    all_pred_boxes = np.array(nms_boxes)
    visualize_bbox(image, all_pred_boxes[..., 1:])
    #later = datetime.now()

    #print(later - now)
    #show_video(model)


def visualize_bbox(img, bboxes, color=config.BOX_COLOR, thickness=2):
    height, width = img.shape[:2]
    """Visualizes a single bounding box on the image"""
    for bbox in bboxes:
        p, x, y, w, h = bbox
        w, h = np.array([w,h])*max(width,height)
        x_min, x_max, y_min, y_max = int(x * width - w / 2), int(x * width + w / 2), int(y * height - h / 2), int(
            y * height + h / 2)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize("car", cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), config.BOX_COLOR, 1)
        cv2.putText(
            img,
            text="car " + str(round(p,2)),
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=config.TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )

    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()
    cv2.imshow("image", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
