from knn import knn_classifier
import pandas as pd

'''
This is the main entry for the models and evaluation. Preprocessing is run beforehand and just has to run once to save on processing time.
'''
def main():
    run_knn()
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



if __name__ == "__main__":
    main()