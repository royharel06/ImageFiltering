import cv2
import numpy as np
import sys
import os
from tensorflow.keras.models import model_from_json

import os
import tensorflow as tf

def load_model():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "DetectingFlares-master")
    json_path = os.path.join(model_dir, "model.json")
    weights_path = os.path.join(model_dir, "model.h5")

    with open(json_path, "r") as json_file:
        json_model = json_file.read()

    model = tf.keras.models.model_from_json(json_model)
    model.load_weights(weights_path)
    return model




def detect_flare_from_image(img_file, model):
    """Calls model.predict(image) for the new image. Prints '1' or '0' for 'faulty' and 'good' images respectively."""
    # pre-processing the image
    new_img = cv2.resize(img_file, (500, 400))
    new_img = np.array(new_img).reshape(-1, 400, 500, 3)

    # inverting output because model was trained to classify the other way.
    if model.predict(new_img) == 0:
        print(1)
    else:
        print(0)


def test_flare_images():
    """Test function to run on flared images"""
    model = load_model()

    DATADIR = "./test/"
    CATEGORY = 'flare'
    path = os.path.join(DATADIR, CATEGORY)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
        except Exception as e:
            print("something's wrong")
        detect_flare_from_image(img_array, model)


def test_good_images():
    """Test function to run on good images"""
    model = load_model()

    DATADIR = "./test/"
    CATEGORY = 'good'
    path = os.path.join(DATADIR, CATEGORY)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
        except Exception as e:
            print("something's wrong")
        detect_flare_from_image(img_array, model)


def main():
    """Main function that runs the classifier over the images provided"""
    # read in images provided
    files = sys.argv[1:]
    # load pre-trained CNN model
    model = load_model()

    # Go through each image and classify
    for file in files:
        img = cv2.imread(str(file))
        detect_flare_from_image(img, model)


if __name__ == '__main__':
    main()