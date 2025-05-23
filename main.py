from knn import knn_classifier
from cnn import train_cnn, predict_images, create_data_generators, plot_cnn_confusion_matrix
from svm import run_svm
from ensemble import get_base_predictions, stack_predictions, train_meta_model, create_images_all_npy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import keras
import numpy as np
import os

'''
This is the main entry for the models and evaluation. Preprocessing is run beforehand and just has to run once to save on processing time.
'''
def main():
    run_knn()
    run_svm()
    run_cnn()
    run_ensemble()
    return

def run_knn():
    '''
    This function runs the KNN classifier for main
    '''
    # Paths to training and test data (WITH HU)
    training_dataset = 'summaries_with_hu/engineered_df_all_with_hu.csv'
    X_test_pca = pd.read_csv('summaries_with_hu/test_engineered_df_with_hu.csv')

    # Load the processed dataset for training (NO HU)
    # training_dataset = 'summaries_no_hu/engineered_df_all_no_hu.csv'
    # X_test_pca = pd.read_csv('summaries_no_hu/test_engineered_df_no_hu.csv')
    
    X_train_pca = pd.read_csv(training_dataset)
    y_train = X_train_pca['ClassId']
    X_train_pca = X_train_pca.drop(columns=['image_path', 'ClassId'])

    knn = knn_classifier(training_dataset)

    # Load the test dataset (NO HU)
    X_test_pca_pred = X_test_pca.drop(columns=['image_path'])
    y_pred = knn.predict(X_test_pca_pred)

    submission_df = pd.DataFrame({'image_path': X_test_pca['image_path'], 'ClassId': y_pred})
    test_metadata = pd.read_csv('testFeatures/test_metadata.csv')
    submission_df = submission_df.merge(test_metadata[['image_path', 'id']], on='image_path', how='inner')
    submission_df = submission_df[['id', 'ClassId']]
    submission_df.to_csv('knn/knn_submission.csv', index=False)
    return

def run_cnn():
    '''
    This function trains and evaluates the CNN model
    '''
    train_dir = '/Users/yuanern/Documents/unimelb/Machine Learning/assignment 2/gtrsb-implementation/train'
    metadata_path = '/Users/yuanern/Documents/unimelb/Machine Learning/assignment 2/gtrsb-implementation/trainFeatures/train_metadata.csv'
    input_shape = (32, 32, 3)
    batch_size = 16
    epochs = 100

    # Create data generators
    train_generator, val_generator, num_classes = create_data_generators(
        train_dir, metadata_path, input_shape[:2], batch_size
    )

    # Train the CNN model
    model = train_cnn(train_generator, val_generator, input_shape, epochs)
    # model = keras.models.load_model('cnn/cnn_model.keras')
    plot_cnn_confusion_matrix(model, val_generator)

    # Prediction
    test_dir = '/Users/yuanern/Documents/unimelb/Machine Learning/assignment 2/gtrsb-implementation/test'
    test_metadata_path = '/Users/yuanern/Documents/unimelb/Machine Learning/assignment 2/gtrsb-implementation/testFeatures/test_metadata.csv'
    predictions = predict_images(model, test_dir, test_metadata_path, input_shape[:2], batch_size)

    class_indices = train_generator.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}

    test_metadata = pd.read_csv(test_metadata_path)

    predicted_indices = predictions.argmax(axis=1)
    predicted_labels = [index_to_class[idx] for idx in predicted_indices]

    submission_df = pd.DataFrame({
        'id': test_metadata['id'],
        'ClassId': predicted_labels
    })

    # Save the DataFrame to a CSV file
    submission_df.to_csv('cnn/cnn_predictions.csv', index=False)
    print("Predictions saved to cnn_predictions.csv")


def run_ensemble():
    """
    Runs the ensemble stacking model by splitting the full engineered data.
    Saves ensemble predictions.
    """
    create_images_all_npy(
        train_dir='train',
        metadata_path='trainFeatures/train_metadata.csv',
        output_path='train_images.npy'
    )

    create_images_all_npy(
        train_dir='test',
        metadata_path='testFeatures/test_metadata.csv',
        output_path='test_images.npy'
    )

    # Loading models and dataset
    knn_model = load('knn/knn_model.joblib')
    cnn_model = keras.models.load_model('cnn/cnn_model.keras')

    df = pd.read_csv('summaries_no_hu/engineered_df_all_no_hu.csv')
    X = df.drop(columns=['image_path', 'ClassId'])
    y = df['ClassId']
    images = np.load('train_images.npy')

    # --- Train/Validation Split ---
    X_train, X_val, y_train, y_val, images_train, images_val = train_test_split(
        X, y, images, test_size=0.2, stratify=y, random_state=42
    )

    # --- Scale validation features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # --- Get base model predictions on validation set ---
    knn_probs_val, cnn_probs_val = get_base_predictions(knn_model, cnn_model, X_val_scaled, images_val)
    stacked_val = stack_predictions(knn_probs_val, cnn_probs_val)

    # --- Train meta-model ---
    meta_model = train_meta_model(stacked_val, y_val)

    # --- Confusion matrix for validation set (last fold) ---
    val_preds = meta_model.predict(stacked_val)
    val_acc = np.mean(val_preds == y_val)
    print(f"Ensemble Validation Accuracy: {val_acc:.4f}")
    all_labels = sorted(y.unique())
    cm = confusion_matrix(y_val, val_preds, labels=all_labels)
    fig, ax = plt.subplots(figsize=(16, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(cmap='Blues', ax=ax, colorbar=True)
    plt.title("Ensemble Confusion Matrix (Validation Set, Last Fold)", fontsize=25)
    ax.set_xlabel("Predicted ClassId", fontsize=15)
    ax.set_ylabel("True ClassId", fontsize=15)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    os.makedirs('ensemble', exist_ok=True)
    plt.savefig('ensemble/ensemble_confusion_matrix_val.png')
    plt.close()
    print("Confusion matrix (validation set) saved to ensemble/ensemble_confusion_matrix_val.png")

    # Load test features with image_path
    test_features = pd.read_csv('summaries_no_hu/test_engineered_df_no_hu.csv')
    test_images = np.load('test_images.npy')
    test_metadata = pd.read_csv('testFeatures/test_metadata.csv')

    # Scale features
    test_features_scaled = scaler.transform(test_features.drop(columns=['image_path']))

    # Get predictions
    knn_test_probs = knn_model.predict_proba(test_features_scaled)
    cnn_test_probs = cnn_model.predict(test_images)
    stacked_test_input = np.hstack([knn_test_probs, cnn_test_probs])
    final_preds = meta_model.predict(stacked_test_input)
    label_encoder = LabelEncoder()
    label_encoder.fit(y) 

    # Inverse transform predictions
    final_labels = label_encoder.inverse_transform(final_preds)

    # Merge with metadata
    submission_df = pd.DataFrame({
        'image_path': test_features['image_path'],
        'ClassId': final_labels
    })
    submission_df = submission_df.merge(test_metadata[['image_path', 'id']], on='image_path', how='inner')
    submission_df = submission_df[['id', 'ClassId']]
    submission_df.to_csv('ensemble/ensemble_predictions.csv', index=False)
    
    return

if __name__ == "__main__":
    main()