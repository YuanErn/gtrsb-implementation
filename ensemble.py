import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from PIL import Image
from cnn import crop_and_enhance

def create_images_all_npy(train_dir='train', metadata_path='trainFeatures/train_metadata.csv', output_path='images_all.npy'):
    """
    Loads all training images, applies crop_and_enhance, and saves as images_all.npy.
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"{output_path} already exists and is not empty.")
        return
    print(f"Creating {output_path} ...")
    metadata = pd.read_csv(metadata_path)
    metadata['image_path'] = metadata['image_path'].apply(lambda x: os.path.join(train_dir, x))
    images = []
    for img_path in metadata['image_path']:
        img = Image.open(img_path)
        arr = crop_and_enhance(img)
        images.append(arr)
    images = np.stack(images)
    np.save(output_path, images)
    print(f"Saved {images.shape[0]} images to {output_path}")

def get_base_predictions(knn_model, cnn_model, val_features, val_images):
    knn_probs = knn_model.predict_proba(val_features)  # Shape: (N, 43)
    cnn_probs = cnn_model.predict(val_images)          # Shape: (N, 43)
    return knn_probs, cnn_probs

# Prediction probabilities from KNN and CNN models stacking
def stack_predictions(knn_probs, cnn_probs):
    return np.hstack([knn_probs, cnn_probs]) 

# Train the meta-model using the stacked predictions
def train_meta_model(stacked_input, y_val):
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(stacked_input, y_val)
    return meta_model


