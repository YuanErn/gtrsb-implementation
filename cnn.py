import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D,
    Flatten, Dense, Dropout, BatchNormalization, Activation
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

def crop_and_enhance(img):
    # Ensure input is 2D or 3D image array
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)  # remove the singleton channel
        elif img.ndim != 2:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        img = Image.fromarray(np.uint8(img))

    elif not isinstance(img, Image.Image):
        raise TypeError(f"Unsupported input type for image: {type(img)}")

    # Crop 10% from each side
    width, height = img.size
    crop_margin_w = int(width * 0.10)
    crop_margin_h = int(height * 0.10)
    img = img.crop((crop_margin_w, crop_margin_h, width - crop_margin_w, height - crop_margin_h))

    img = img.resize((32, 32))
    
    # Enhance
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Brightness(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.3)

    # Convert to grayscale and normalize
    img = img.convert('L')
    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0  # Normalize here just in case
    arr = np.expand_dims(arr, axis=-1)  # shape (H, W, 1)
    return arr


def create_data_generators(train_dir, metadata_path, input_shape=(64, 64), batch_size=32, validation_split=0.2, seed=42):
    """
    Creates training and validation data generators. This preprocessing will be applied to the images for
    training.
    """
    metadata = pd.read_csv(metadata_path)
    metadata['image_path'] = metadata['image_path'].apply(lambda x: os.path.join(train_dir, x))
    metadata['ClassId'] = metadata['ClassId'].astype(str)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        zoom_range=0.1,
        validation_split=validation_split,
        horizontal_flip=False,
    )

    train_generator = datagen.flow_from_dataframe(
        metadata,
        x_col='image_path',
        y_col='ClassId',
        target_size=input_shape,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=seed,
        preprocessing_function=crop_and_enhance
    )

    val_generator = datagen.flow_from_dataframe(
        metadata,
        x_col='image_path',
        y_col='ClassId',
        target_size=input_shape,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=seed,
        preprocessing_function=crop_and_enhance
    )

    num_classes = len(train_generator.class_indices)
    return train_generator, val_generator, num_classes

def train_cnn(train_generator, val_generator, input_shape=(64, 64, 3), epochs=25):
    """
    Trains a CNN model using images and metadata.
    """
    num_classes = len(train_generator.class_indices)

    # Calculate class weights
    class_labels = list(train_generator.class_indices.values())
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(class_labels),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    model = Sequential()

    # Input: 32x32x1
    # Block 1
    model.add(Conv2D(32, (5,5), padding='same', input_shape=(32,32,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Block 2
    model.add(Conv2D(64, (5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # MaxPool 1
    model.add(MaxPooling2D(pool_size=(3,3), strides=2))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Block 4
    model.add(Conv2D(30, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # MaxPool 2
    model.add(MaxPooling2D(pool_size=(3,3), strides=2))
    model.add(Dropout(0.25))

    # Block 5
    model.add(Conv2D(30, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Dense Layers
    model.add(Flatten())
    model.add(Dense(600))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(300))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(43, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with class weights
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=[early_stop]
    )   

    # Save the trained model
    os.makedirs('cnn', exist_ok=True)
    model.save('cnn/cnn_model.keras')
    print("Model saved as cnn_model.keras")

    # Loss curve
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('cnn/cnn_loss_curve.png')
    plt.close()
    print("Loss curve saved to cnn/cnn_loss_curve.png")

    # Accuracy curve
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('cnn/cnn_accuracy_curve.png')
    plt.close()
    print("Accuracy curve saved to cnn/cnn_accuracy_curve.png")


    return model

def predict_images(model, image_dir, metadata_path, input_shape=(64, 64), batch_size=32):
    """
    Generates predictions for images using a trained model.
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata['image_path'] = metadata['image_path'].apply(lambda x: os.path.join(image_dir, x))

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        zoom_range=0.1,
        horizontal_flip=False,
        preprocessing_function=crop_and_enhance
    )

    prediction_generator = datagen.flow_from_dataframe(
        dataframe=metadata,
        x_col='image_path',
        y_col=None,
        target_size=input_shape,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    predictions = model.predict(prediction_generator)
    return predictions
