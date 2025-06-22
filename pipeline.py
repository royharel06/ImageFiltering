import sys
import os
import warnings
import numpy as np
import pandas as pd
import cv2
import shutil

# Add src folder to path manually
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Image-Quality-Assessment", "src")))
sys.path.append(os.path.abspath("horizon_detection"))
sys.path.append(os.path.abspath("DetectingFlares-master"))

from inference import get_image_quality_scores
from horizon_demo import sat_horizon_present
from detector import load_model


# === From content-filter.py ===
def is_mostly_space(img_path, dark_thresh=15, ratio_thresh=0.6):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(gray < dark_thresh)
    total_pixels = gray.size
    ratio = dark_pixels / total_pixels
    print(f"[SPACE] Dark ratio: {ratio:.2f}")
    return ratio > ratio_thresh

def is_mostly_water(img_path, blue_thresh=1.15, ratio_thresh=0.6):
    img = cv2.imread(img_path)
    b, g, r = cv2.split(img)
    blue_dominant = (b > r * blue_thresh) & (b > g * blue_thresh)
    ratio = np.sum(blue_dominant) / (img.shape[0] * img.shape[1])
    print(f"[WATER] Blue dominant ratio: {ratio:.2f}")
    return ratio > ratio_thresh

# === Flare Detection Function ===
def detect_flare_from_image(img, model):
    new_img = cv2.resize(img, (500, 400)).reshape(-1, 400, 500, 3)
    return model.predict(new_img)[0][0] == 0  # True if flare

# === Main Pipeline ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow INFO & WARNING
warnings.filterwarnings("ignore")         # suppress sklearn and other warnings

input_dir = "input"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
flare_model = load_model()

results = []

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)
    if img is None:
        results.append({"image": filename, "status": "Error loading"})
        continue
    
    horizon = sat_horizon_present(img)
    quality = get_image_quality_scores(path)
    blur_score = quality["score"]

    # Space filter
    if is_mostly_space(path):
        results.append({"image": filename, "status": "Dropped at space filter"})
        print("\n=== Image Dropped - Space ===")
        print(f"Image name: {filename}")
        print(f"Blur score: {blur_score:.2f}")
        print(f"Horizon present: {horizon}")
        print("\n\n")
        continue

    # Water filter
    if is_mostly_water(path):
        results.append({"image": filename, "status": "Dropped at water filter"})
        print("\n=== Image Dropped - Water ===")
        print(f"Image name: {filename}")
        print(f"Blur score: {blur_score:.2f}")
        print(f"Horizon present: {horizon}")
        print("\n\n")
        continue

    # Flare
    if detect_flare_from_image(img, flare_model):
        results.append({"image": filename, "status": "Dropped at flare filter"})
        print("\n=== Image Dropped - Flare ===")
        print(f"Image name: {filename}")
        print(f"Blur score: {blur_score:.2f}")
        print(f"Horizon present: {horizon}")
        print("\n\n")
        continue
    
    # Blur
    if blur_score > 80:
        results.append({"image": filename, "status": "Dropped at blur filter", "blur_score": blur_score})
        print("\n=== Image Dropped - Blur ===")
        print(f"Image name: {filename}")
        print(f"Blur score: {blur_score:.2f}")
        print(f"Horizon present: {horizon}")
        print("\n\n")
        continue

    # Passed all
    shutil.copy(path, os.path.join(output_dir, filename))
    results.append({
        "image": filename,
        "status": "Passed",
        "blur_score": blur_score,
        "horizon_present": horizon
    })
    
    print("\n=== Image Passed ===")
    print(f"Image name: {filename}")
    print(f"Blur score: {blur_score:.2f}")
    print(f"Horizon present: {horizon}")
    print("\n\n")
