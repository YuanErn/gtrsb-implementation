from knn import knn_classifier
from cnn import train_cnn, predict_images, create_data_generators
import pandas as pd

'''
This is the main entry for the models and evaluation. Preprocessing is run beforehand and just has to run once to save on processing time.
'''
def main():
    run_cnn()
    return

def run_knn():
    '''
    This function runs the KNN classifier for main
    '''
    # Load the processed dataset for training
    training_dataset = 'engineered_df_all.csv'
    X_train_pca = pd.read_csv(training_dataset)
    y_train = X_train_pca['ClassId']
    X_train_pca = X_train_pca.drop(columns=['image_path', 'ClassId'])

    knn = knn_classifier(training_dataset)

    # Load the test dataset
    X_test_pca = pd.read_csv('test_engineered_df.csv')
    X_test_pca_pred = X_test_pca.drop(columns=['image_path'])
    # Make predictions on the test set and save the results
    y_pred = knn.predict(X_test_pca_pred)

    submission_df = pd.DataFrame({'image_path': X_test_pca['image_path'], 'ClassId': y_pred})
    train_metadata = pd.read_csv('testFeatures/test_metadata.csv')
    submission_df = submission_df.merge(train_metadata[['image_path', 'id']], on='image_path', how='inner')
    submission_df = submission_df[['id', 'ClassId']]
    submission_df.to_csv('knn/submission.csv', index=False)
    return

def run_cnn():
    '''
    This function trains and evaluates the CNN model
    '''
    train_dir = '/Users/yuanern/Documents/unimelb/Machine Learning/assignment 2/gtrsb-implementation/train'
    metadata_path = '/Users/yuanern/Documents/unimelb/Machine Learning/assignment 2/gtrsb-implementation/trainFeatures/train_metadata.csv'
    input_shape = (64, 64, 3)
    batch_size = 32
    epochs = 100

    # Create data generators
    train_generator, val_generator, num_classes = create_data_generators(
        train_dir, metadata_path, input_shape[:2], batch_size
    )

    # Train the CNN model
    model = train_cnn(train_generator, val_generator, input_shape, epochs)

    # Example prediction (optional)
    test_dir = '/Users/yuanern/Documents/unimelb/Machine Learning/assignment 2/gtrsb-implementation/test'
    test_metadata_path = '/Users/yuanern/Documents/unimelb/Machine Learning/assignment 2/gtrsb-implementation/testFeatures/test_metadata.csv'
    predictions = predict_images(model, test_dir, test_metadata_path, input_shape[:2], batch_size)

    # Load test metadata to get the IDs
    test_metadata = pd.read_csv(test_metadata_path)

    # Create a DataFrame with 'id' and 'ClassId'
    submission_df = pd.DataFrame({
        'id': test_metadata['id'],
        'ClassId': predictions.argmax(axis=1)  # Convert probabilities to class indices
    })

    # Save the DataFrame to a CSV file
    submission_df.to_csv('cnn/cnn_predictions.csv', index=False)
    print("Predictions saved to cnn_predictions.csv")

if __name__ == "__main__":
    main()