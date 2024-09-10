from random import random


import cv2
import torchvision.transforms as T

def plot_one_box(x, img, color=None, label=None, line_thickness=None, score=None):
    # Plots one bounding box on image img, with optional score (similarity)

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        if score is not None:
            label_with_score = f"{label}: {score:.2f}"  # Format label with score
        else:
            label_with_score = label
        t_size = cv2.getTextSize(label_with_score, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label_with_score, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def build_transforms(cfg):
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        normalize_transform
    ])

    return transform
