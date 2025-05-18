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
import keras

def create_data_generators(train_dir, metadata_path, input_shape=(64, 64), batch_size=32, validation_split=0.2, seed=42):
    """
    Creates training and validation data generators. This preprocessing will be applied to the images for
    training.
    """
    metadata = pd.read_csv(metadata_path)
    metadata['image_path'] = metadata['image_path'].apply(lambda x: os.path.join(train_dir, x))
    metadata['ClassId'] = metadata['ClassId'].astype(str)  # Ensure labels are strings

    # Initialize ImageDataGenerator with validation split 
    datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    zoom_range=0.1,
    validation_split=validation_split
    )

    # Create training data generator
    train_generator = datagen.flow_from_dataframe(
        metadata,
        x_col='image_path',
        y_col='ClassId',
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=seed
    )

    # Create validation data generator
    val_generator = datagen.flow_from_dataframe(
        metadata,
        x_col='image_path',
        y_col='ClassId',
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=seed
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

    # Define the CNN model
    model = Sequential()
    
    # First block: 32 filters @ 3×3 (two Convs + Activations), then Pool + Dropout
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Next blocks: doubling filters each time (64, 128, 256…)
    for filters in [64, 128, 256]:
        model.add(Conv2D(filters, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
    
    # Flatten → Dense head → final softmax
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    # Compile the model
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, weight_decay=1e-6)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with class weights
    model.fit(
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

    return model

def predict_images(model, image_dir, metadata_path, input_shape=(64, 64), batch_size=32):
    """
    Generates predictions for images using a trained model.
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata['image_path'] = metadata['image_path'].apply(lambda x: os.path.join(image_dir, x))

    # Data generator for prediction
    datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    zoom_range=0.1,
    validation_split=0.2
    )

    prediction_generator = datagen.flow_from_dataframe(
        dataframe=metadata,
        x_col='image_path',
        y_col=None,
        target_size=input_shape,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    # Generate predictions

    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    predictions = model.predict(prediction_generator)

    return predictions
