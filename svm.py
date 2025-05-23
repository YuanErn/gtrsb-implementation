import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def run_svm():
    # Paths to training and test data (NO HU)
    # train_df_path = 'summaries_no_hu/engineered_df_all_no_hu.csv'
    # test_df_path = 'summaries_no_hu/test_engineered_df_no_hu.csv'
    # test_metadata_path = 'testFeatures/test_metadata.csv'

    # Paths to training and test data (WITH HU)
    train_df_path = 'summaries_with_hu/engineered_df_all_with_hu.csv'
    test_df_path = 'summaries_with_hu/test_engineered_df_with_hu.csv'
    test_metadata_path = 'testFeatures/test_metadata.csv'

    # Load training data
    train_df = pd.read_csv(train_df_path)
    X = train_df.drop(columns=['image_path', 'ClassId'])
    y = train_df['ClassId']

    # Defining hyperparameters
    C = 10
    gamma = 0.001

    # K-Fold Cross-Validation
    k = 10
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    val_accuracies = []
    val_reports = []

    print(f"Running {k}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = SVC(kernel='rbf', C=C, gamma=gamma)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_val_pred)
        val_accuracies.append(acc)
        val_reports.append(classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)) 
        print(f"Fold {fold} Validation Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_val, y_val_pred)
    fig, ax = plt.subplots(figsize=(16, 16)) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', ax=ax, colorbar=True)
    plt.title("SVM Confusion Matrix (Last Fold)", fontsize=25)
    ax.set_xlabel("Predicted ClassId", fontsize=15)
    ax.set_ylabel("True ClassId", fontsize=15)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig("svm/svm_confusion_matrix_kfold.png")
    plt.close()
    print(f"\nMean Validation Accuracy: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")

    weighted_metrics = ['precision', 'recall', 'f1-score']
    weighted_avgs = {metric: [] for metric in weighted_metrics}

    for report in val_reports:
        for metric in weighted_metrics:
            weighted_avgs[metric].append(report['weighted avg'][metric])

    print("\nAverage Validation Classification Report (weighted avg):")
    for metric in weighted_metrics:
        avg = np.mean(weighted_avgs[metric])
        print(f"{metric.capitalize()}: {avg:.4f}")

    # Train on full data for test prediction
    model = SVC(kernel='rbf', C=C, gamma=gamma)
    model.fit(X, y)

    # Load test data and merge with metadata to get 'id'
    test_data = pd.read_csv(test_df_path)
    test_metadata = pd.read_csv(test_metadata_path)
    test_data = test_data.merge(test_metadata[['image_path', 'id']], on='image_path', how='inner')

    # Extract features (X_test) and IDs
    X_test = test_data.drop(columns=['image_path', 'id'])
    test_ids = test_data['id']

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Create a DataFrame with 'id' and 'ClassId'
    submission_df = pd.DataFrame({
        'id': test_ids,
        'ClassId': y_pred
    })

    os.makedirs('svm', exist_ok=True)
    submission_df.to_csv('svm/svm_predictions.csv', index=False)
    print("Predictions saved to svm/svm_predictions.csv")

    return

def svm_grid_search(X_train, X_val, y_train, y_val, C_values, gamma_values):
    """
    Performs grid search over C and gamma, returns accuracy matrix.
    """
    accuracy_matrix = np.zeros((len(C_values), len(gamma_values)))
    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            model = SVC(kernel='rbf', C=C, gamma=gamma)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_val, model.predict(X_val))
            accuracy_matrix[i, j] = acc
    return accuracy_matrix

def plot_svm_grid_search(C_values, gamma_values, accuracy_matrix):
    """
    Plots a heatmap of the accuracy matrix for C and gamma.
    """
    # Make gamma values slightly smaller (e.g., divide by 10)
    gamma_values = [g / 10 for g in gamma_values]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(accuracy_matrix, interpolation='nearest', cmap='viridis')
    plt.title("Validation Accuracy for SVM Grid Search")
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.colorbar(im, label="Accuracy")

    plt.xticks(np.arange(len(gamma_values)), [f"{g:.1e}" for g in gamma_values], rotation=90)
    plt.yticks(np.arange(len(C_values)), [f"{c:.1e}" for c in C_values])

    plt.tight_layout()
    plt.savefig('svm/svm_grid_search_heatmap.png')
    plt.show()
    print("Heatmap saved to svm/svm_grid_search_heatmap.png")

