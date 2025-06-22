import cv2
import os
import numpy as np

def sat_horizon_present(img, brightness_diff_thresh=80):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    max_diff = 0
    best_split = -1

    # Try splits from 20% to 80% of image height
    for split in range(int(0.2 * h), int(0.8 * h), int(h * 0.05)):
        top = gray[:split, :]
        bottom = gray[split:, :]
        if top.size == 0 or bottom.size == 0:
            continue
        top_mean = top.mean()
        bot_mean = bottom.mean()
        diff = abs(top_mean - bot_mean)
        if diff > max_diff:
            max_diff = diff
            best_split = split

    #print(f"Max brightness diff: {max_diff:.2f} at row {best_split}")
    return max_diff > brightness_diff_thresh


def test_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Could not read image: {path}")
        return
    result = sat_horizon_present(img)
    status = "Horizon Detected" if result else "No Horizon"
    #print(f"{path}: {status}")

if __name__ == "__main__":
    # Change this to a folder or list of satellite images
    test_images = [
        "B3_ER_640x480_46.jpg",
        "B3_HZ_640x480_35.jpg"
    ]

    for img_path in test_images:
        test_image(img_path)
