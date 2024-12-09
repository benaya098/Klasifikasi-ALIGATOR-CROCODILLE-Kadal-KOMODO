import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class ImageDataLoader:
    def __init__(self, train_dir, test_dir, img_size=110):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.labels = ["alligator", "crocodile", "komodo", "kadal"]
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}

    def load_data(self):
        train_images = []
        train_labels = []

        for filename in os.listdir(self.train_dir):
            img_path = os.path.join(self.train_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (self.img_size, self.img_size))
                train_images.append(img)

                label_found = False
                for label in self.labels:
                    if label in filename.lower():
                        train_labels.append(self.label_map[label])
                        label_found = True
                        break
                if not label_found:
                    print(f"Label not found for image: {filename}")

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
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
            else:
                print(f"Error loading image: {filename}")

        test_images = np.array(test_images)
        return test_images, filenames
