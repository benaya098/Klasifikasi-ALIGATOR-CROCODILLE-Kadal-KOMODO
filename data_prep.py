import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 110  # Ukuran gambar (110x110 pixel)
LABELS = ["alligator", "crocodile", "komodo", "kadal"]  # Label kelas

class ImageDataLoader:
    def __init__(self, train_dir, test_dir, img_size=IMG_SIZE):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size

    def load_data(self):
        train_images = []
        train_labels = []

        for filename in os.listdir(self.train_dir):
            img_path = os.path.join(self.train_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.img_size, self.img_size))
                train_images.append(img)

                # Menentukan label berdasarkan nama file
                for idx, label in enumerate(LABELS):
                    if label in filename:
                        train_labels.append(idx)
                        break

        # Normalisasi data
        train_images = np.array(train_images) / 255.0
        train_labels = np.array(train_labels)

        # Split data ke training dan validation
        return train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    def load_test_data(self):
        test_images = []
        filenames = []

        for filename in os.listdir(self.test_dir):
            img_path = os.path.join(self.test_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.img_size, self.img_size))
                test_images.append(img)
                filenames.append(filename)

        return np.array(test_images) / 255.0, filenames
