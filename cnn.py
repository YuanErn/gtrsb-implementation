import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def crop_and_enhance(img, save_cropped=False, save_dir="cropped_samples", original_path=None):
    # Ensure input is 2D or 3D image array
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1) 
        elif img.ndim != 3:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        img = Image.fromarray(np.uint8(img))
    elif not isinstance(img, Image.Image):
        raise TypeError(f"Unsupported input type for image: {type(img)}")

    # Crop from each side
    width, height = img.size
    crop_margin_w = int(width * 0.1)
    crop_margin_h = int(height * 0.1)
    img = img.crop((crop_margin_w, crop_margin_h, width - crop_margin_w, height - crop_margin_h))

    img = img.resize((32, 32))

    # Enhance brightness and sharpness and color correct
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.3)  # Boost color saturation

    arr = np.array(img).astype(np.float32)
    arr[..., 0] *= 1.4  # Boost red channel
    arr = np.clip(arr, 0, 255)
    img = Image.fromarray(arr.astype(np.uint8))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2) 
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.5)

    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0

    # Save cropped image if requested
    if save_cropped and original_path is not None:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(original_path)  
        filename = f"crop_{base_name}"             
        img.save(os.path.join(save_dir, filename))

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
        validation_split=validation_split,
        horizontal_flip=False,
    )

    train_generator = datagen.flow_from_dataframe(
        metadata,
        x_col='image_path',
        y_col='ClassId',
        target_size=input_shape,
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=seed,
        shuffle=True,
        preprocessing_function=crop_and_enhance
    )

    val_generator = datagen.flow_from_dataframe(
        metadata,
        x_col='image_path',
        y_col='ClassId',
        target_size=input_shape,
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=seed,
        shuffle=False,
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
    model.add(Conv2D(32, (5,5), padding='same', input_shape=(32,32,3)))
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

def predict_images(model, image_dir, metadata_path, input_shape=(32, 32), batch_size=32):
    """
    Generates predictions for images using a trained model.
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata['image_path'] = metadata['image_path'].apply(lambda x: os.path.join(image_dir, x))

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=False,
    )

    prediction_generator = datagen.flow_from_dataframe(
        dataframe=metadata,
        x_col='image_path',
        y_col=None,
        target_size=input_shape,
        color_mode='rgb',  
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
        preprocessing_function=crop_and_enhance
    )


    predictions = model.predict(prediction_generator)
    return predictions


def plot_cnn_confusion_matrix(model, val_generator, save_path="cnn/cnn_confusion_matrix_val.png"):
    """
    Plots and saves the confusion matrix for the CNN on the validation set.
    """
    # Get true labels
    y_true = val_generator.classes

    class_indices = val_generator.class_indices
    sorted_labels = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

    # Get predicted probabilities
    y_pred_probs = model.predict(val_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted_labels)    
    disp.plot(cmap='Blues', ax=ax, colorbar=True)
    plt.title("CNN Confusion Matrix (Validation Set, Best Epoch)", fontsize=25)
    ax.set_xlabel("Predicted ClassId", fontsize=15)
    ax.set_ylabel("True ClassId", fontsize=15)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    
    return 
