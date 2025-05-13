from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def knn_classifier(csv_path, k=1, n_splits=10):
    """
    Runs KNN classification with Stratified K-Fold cross-validation.
    Saves confusion matrix of last fold and overall average metrics.
    """
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['image_path', 'ClassId'])
    y = df['ClassId']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []
    reports = []
    last_cm = None

    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        reports.append(classification_report(y_test, y_pred, output_dict=True, zero_division=0))       
        last_cm = confusion_matrix(y_test, y_pred)

        print(f"Fold {fold} accuracy: {acc:.4f}")
        fold += 1

    # Save confusion matrix from the last fold
    os.makedirs('knn', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(last_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Last Fold)')
    plt.colorbar()
    labels = sorted(y.unique())
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('knn/knn_confusion_matrix_kfold.png')
    print("Confusion matrix (last fold) saved to knn_confusion_matrix_kfold.png")

    # Compute average accuracy and classification report
    avg_acc = np.mean(accuracies)
    # Convert list of dicts to list of DataFrames
    report_dfs = [pd.DataFrame(r).transpose() for r in reports]
    # Align them and take the mean
    avg_report = pd.concat(report_dfs).groupby(level=0).mean()
    

    # Save evaluation results
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/knn_kfold_eval.txt', 'w') as f:
        f.write(f"Average Accuracy: {avg_acc:.4f}\n\n")
        f.write("Average Classification Report:\n")
        f.write(avg_report.to_string())
    print("K-Fold evaluation results saved to metrics/knn_kfold_eval.txt")
    return knn

def tune_k_for_knn(csv_path, k_range=range(1, 21), n_splits=10):
    """
    Loops over different k values for KNN, performs cross-validation,
    stores the average accuracy, and plots k vs accuracy.
    """
    
    accuracies = []

    for k in k_range:
        print(f"\nEvaluating KNN with k={k}...")
        knn = knn_classifier(csv_path, k=k, n_splits=n_splits)
        
        # Read back the accuracy from the saved file (or recompute here)
        with open('knn/knn_kfold_eval.txt', 'r') as f:
            lines = f.readlines()
            avg_acc_line = next((line for line in lines if line.startswith("Average Accuracy")), None)
            if avg_acc_line:
                acc = float(avg_acc_line.strip().split(":")[1])
                accuracies.append(acc)
            else:
                print(f"Warning: Accuracy not found for k={k}")
                accuracies.append(0)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), accuracies, marker='o', color='blue')
    plt.title("KNN Hyperparameter Tuning (k vs Accuracy)")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Average Cross-Validation Accuracy")
    plt.xticks(list(k_range))
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("knn", exist_ok=True)
    plt.savefig("knn/k_optimise.png")
    print("k optimisation plot saved to knn/k_optimise.png")

    # Return the best k and its accuracy
    best_k = k_range[np.argmax(accuracies)]
    best_acc = max(accuracies)
    print(f"\nBest k: {best_k} with average accuracy: {best_acc:.4f}")
    return best_k, best_acc

# Used for tuning k
# best_k, best_acc = tune_k_for_knn("engineered_df_all.csv", k_range=range(1, 21))
