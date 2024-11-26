import os
import cv2
import numpy as np
from random import shuffle
import constants as CONST

def label_img(name):
    """
    Function to determine the label from the file name.
    Arguments:
    - name: Image file name, e.g., 'crocodile.123.jpg'
    
    Return:
    - label_arr: Binary label array [1, 0, 0, 0] for crocodile, [0, 1, 0, 0] for alligator, etc.
    """
    word_label = name.split('.')[0]  # Take the first part of the file name
    if word_label in CONST.LABEL_MAP:
        label = CONST.LABEL_MAP[word_label]
        label_arr = np.zeros(4)  #
        label_arr[label] = 1
        return label_arr
    else:
        print(f"Label for file '{name}' not found in LABEL_MAP.")
        return None

def prep_and_load_data():
    """
    Function to load and process image data from the `train` folder.
    - Resizes images according to IMG_SIZE.
    - Labels images based on file names.
    - Performs pixel normalization.

    Return:
    - images: Numpy array of processed images.
    - labels: Numpy array of labels for each image.
    """
    DIR = CONST.TRAIN_DIR  # Get train folder path from constants.py
    data = []
    image_paths = os.listdir(DIR)  # Get all image files in the train folder
    shuffle(image_paths)  # Randomize file order

    count = 0
    for img_path in image_paths:
        label = label_img(img_path)  # Get label from file name
        if a label is None:
            continue  # Skip if label is not valid

        # Process image
        path = os.path.join(DIR, img_path)
        image = cv2.imread(path)  # Read image using OpenCV
        if image is None:
            print(f"Image {img_path} cannot be read, skipped.")
            continue

        image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))  # Resize image to 110x110
        image = image.astype('float32') / 255.0  # Normalize pixel values to range [0, 1]
        data.append([image, label])  # Add image and label to data list

        count += 1
        if count == CONST.DATA_SIZE:  # If DATA_SIZE limit is reached, stop the loop
            break

    # Shuffle data before using for training
    shuffle(data)

    # Separate images and labels into separate arrays
    images = np.array([i[0] for i in data], dtype=np.float32)
    labels = np.array([i[1] for i in data], dtype=np.float32)

    print(f"Total Samples Loaded : {len(data)}")
    print(f"Image shape : {images.shape}")
    print(f"Labels shape : {labels.shape}")

    return images, labels

