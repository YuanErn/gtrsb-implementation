import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def run_svm():
    # Paths to training and test data
    train_df_path = 'engineered_df_all.csv'  # PCA-reduced training data
    test_df_path = 'test_engineered_df.csv'  # PCA-reduced test data
    test_metadata_path = 'testFeatures/test_metadata.csv'

    # Load training data
    X = pd.read_csv(train_df_path).drop(columns=['image_path', 'ClassId'])
    y = pd.read_csv(train_df_path)['ClassId']

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load test data and merge with metadata to get 'id'
    test_data = pd.read_csv(test_df_path)
    test_metadata = pd.read_csv(test_metadata_path)
    test_data = test_data.merge(test_metadata[['image_path', 'id']], on='image_path', how='inner')

    # Extract features (X_test) and IDs
    X_test = test_data.drop(columns=['image_path', 'id'])
    test_ids = test_data['id']

    # Train the SVM
    model = SVC(kernel='rbf', C=10, gamma=0.1)
    model.fit(X_train, y_train)

    # Evaluate on the validation set
    y_val_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Create a DataFrame with 'id' and 'ClassId'
    submission_df = pd.DataFrame({
        'id': test_ids,  
        'ClassId': y_pred
    })

    # Ensure the output directory exists then save
    os.makedirs('svm', exist_ok=True)
    submission_df.to_csv('svm/svm_predictions.csv', index=False)
    print("Predictions saved to svm/svm_predictions.csv")