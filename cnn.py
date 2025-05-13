import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

def train_cnn(train_dir, metadata_path, input_shape=(64, 64, 3), batch_size=32, epochs=10):
    """
    Trains a CNN model using images and metadata.

    Args:
        train_dir (str): Path to the directory containing training images.
        metadata_path (str): Path to the CSV file containing metadata with image paths and labels.
        input_shape (tuple): Shape of the input images (default is (64, 64, 3)).
        batch_size (int): Batch size for training (default is 32).
        epochs (int): Number of epochs to train the model (default is 10).

    Returns:
        model: The trained CNN model.
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata['image_path'] = metadata['image_path'].apply(lambda x: os.path.join(train_dir, x))

    # Data generators for training and validation
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize pixel values
        validation_split=0.2  # Split 20% of data for validation
    )

    train_generator = datagen.flow_from_dataframe(
        metadata,
        x_col='image_path',
        y_col='ClassId',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )

    val_generator = datagen.flow_from_dataframe(
        metadata,
        x_col='image_path',
        y_col='ClassId',
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    # Define the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # Save the trained model
    model.save('cnn_model.h5')
    print("Model saved as cnn_model.h5")

    return model

# Example usage
if __name__ == "__main__":
    train_dir = '/Users/yuanern/Documents/unimelb/Machine Learning/assignment 2/2025_A2/train'
    metadata_path = '/Users/yuanern/Documents/unimelb/Machine Learning/assignment 2/2025_A2/trainFeatures/train_metadata.csv'
    train_cnn(train_dir, metadata_path)