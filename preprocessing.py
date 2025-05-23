import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import warnings
import os
import cv2

# ------------------- Feature Extraction Functions -------------------

def extract_hu_moments(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros(7)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    else:
        hu_moments = np.zeros(7)
    return hu_moments

def create_hu_moments_df(metadata_df, image_root_dir):
    hu_features = []
    for img_rel_path in metadata_df['image_path']:
        img_full_path = os.path.join(image_root_dir, img_rel_path)
        hu = extract_hu_moments(img_full_path)
        hu_features.append(hu)
    hu_df = pd.DataFrame(hu_features, columns=[f'hu_{i+1}' for i in range(7)])
    hu_df['image_path'] = metadata_df['image_path'].values
    return hu_df

def merge_datasets(metadata_path, hog_path, color_hist_path, additional_feat_path):
    train_metadata = pd.read_csv(metadata_path)
    hog_pca = pd.read_csv(hog_path)
    color_histogram = pd.read_csv(color_hist_path)
    additional_features = pd.read_csv(additional_feat_path)
    merged = train_metadata \
        .merge(hog_pca, on='image_path', how='inner') \
        .merge(color_histogram, on='image_path', how='inner') \
        .merge(additional_features, on='image_path', how='inner')
    merged_df = merged.loc[:, ~merged.columns.duplicated()]
    return merged_df

# ------------------- Visualization Functions -------------------

def plot_class_distribution(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        df['ClassId'],
        bins=range(int(df['ClassId'].min()), int(df['ClassId'].max()) + 2),
        color="#6A5ACD",
        edgecolor="black",
        ax=ax
    )
    ax.set_title("Distribution of Traffic Sign Classes", fontsize=16, weight='bold')
    ax.set_xlabel("Traffic Sign Class ID", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()

def plot_pca_loadings_heatmap(loadings_df, save_path):
    fig_width = min(20, 0.1 * loadings_df.shape[1])
    fig_height = min(10, 0.4 * loadings_df.shape[0])
    plt.figure(figsize=(fig_width, fig_height))
    plt.xticks(rotation=90, fontsize=4)
    sns.heatmap(loadings_df, cmap="coolwarm", center=0, annot=False,
                cbar_kws={'label': 'Feature Loading'})
    plt.title("PCA Component Loadings Heatmap", fontsize=16, weight='bold')
    plt.xlabel("Original Features")
    plt.ylabel("Principal Components")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ------------------- PCA and Feature Engineering -------------------

def PCA_transform(dataframe, variance_exp):
    X = dataframe.drop(columns=['image_path', 'ClassId', 'id'], errors='ignore')
    mask_finite = np.isfinite(X).all(axis=0)
    X = X.loc[:, mask_finite]
    feature_names = list(X.columns)
    sel = VarianceThreshold(threshold=0.0)
    sel.fit(X)
    keep_mask = sel.get_support()
    X = X.loc[:, keep_mask]
    feature_names = [f for f, keep in zip(feature_names, keep_mask) if keep]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pca = PCA(svd_solver='auto')
        pca.fit(X_scaled)
    cum_var = pca.explained_variance_ratio_.cumsum()
    n_components = (cum_var < variance_exp).sum() + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pca = PCA(svd_solver = 'auto', n_components=n_components)
        pca.fit_transform(X_scaled)
    loadings = pca.components_
    var_ratios = pca.explained_variance_ratio_
    feature_contributions = np.sum((loadings ** 2) * var_ratios[:, np.newaxis], axis=0)
    feature_contributions /= feature_contributions.sum()
    ranked_df = pd.DataFrame({
        'Feature': feature_names,
        'Variance Contribution': feature_contributions
    }).sort_values(by='Variance Contribution', ascending=False).reset_index(drop=True)
    ranked_df['Cumulative'] = ranked_df['Variance Contribution'].cumsum()
    selected_features = ranked_df[ranked_df['Cumulative'] <= variance_exp]['Feature'].tolist()
    if ranked_df['Cumulative'].iloc[len(selected_features)] < variance_exp:
        selected_features.append(ranked_df['Feature'].iloc[len(selected_features)])
    loadings_df = pd.DataFrame(
        loadings,
        columns=feature_names,
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    variance_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(explained_var))],
        'ExplainedVariance': explained_var,
        'CumulativeVariance': cumulative_var
    })
    return ranked_df, variance_df, selected_features, loadings_df, pca, scaler

def generate_pca_weighted_features(dataframe, loadings_df, selected_pcs, loading_threshold=0.1):
    if selected_pcs is None:
        selected_pcs = loadings_df.index.tolist()
    engineered_features = {}
    for pc in selected_pcs:
        if pc not in loadings_df.index:
            print(f"Warning: {pc} not found in loadings_df")
            continue
        pc_loadings = loadings_df.loc[pc]
        selected_features = pc_loadings[pc_loadings.abs() >= loading_threshold]
        if selected_features.empty:
            print(f"No features meet threshold for {pc}")
            continue
        available_features = [f for f in selected_features.index if f in dataframe.columns]
        weights = selected_features[available_features]
        if weights.empty:
            print(f"No matching features found in dataframe for {pc}")
            continue
        engineered_features[pc + '_weighted'] = dataframe[weights.index] @ weights.values
    return pd.DataFrame(engineered_features)

def preprocess_test_dataset(test_metadata_path, hog_path, color_hist_path, additional_feat_path, pca_model, loading_df, selected_pcs, scaler_model, output_path):
    merged_df = merge_datasets(
        test_metadata_path,
        hog_path,
        color_hist_path,
        additional_feat_path
    )
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    X_test = merged_df.drop(columns=['image_path'], errors='ignore')
    X_test = X_test.loc[:, loading_df.columns]
    X_test_scaled = scaler_model.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    engineered_features_df = generate_pca_weighted_features(
        dataframe=X_test_scaled_df,
        loadings_df=loading_df,
        selected_pcs=selected_pcs,
        loading_threshold=0.1
    )
    engineered_features_df = pd.concat([merged_df[['image_path']], engineered_features_df], axis=1)
    engineered_features_df.to_csv(output_path, index=False)
    return

# ------------------- Pipeline Functions -------------------

def save_hu_features():
    # For train
    train_metadata = pd.read_csv('trainFeatures/train_metadata.csv')
    hu_moments_df = create_hu_moments_df(train_metadata, 'train')
    additional_features = pd.read_csv('trainFeatures/additional_features.csv')
    # Drop any existing hu_* columns
    hu_cols = [col for col in additional_features.columns if col.startswith('hu_')]
    additional_features = additional_features.drop(columns=hu_cols, errors='ignore')
    additional_features_with_hu = additional_features.merge(hu_moments_df, on='image_path', how='left')
    additional_features_with_hu.to_csv('trainFeatures/additional_features_with_hu.csv', index=False)
    # For test
    test_metadata = pd.read_csv('testFeatures/test_metadata.csv')
    hu_moments_test_df = create_hu_moments_df(test_metadata, 'test')
    additional_features_test = pd.read_csv('testFeatures/additional_features.csv')
    hu_cols_test = [col for col in additional_features_test.columns if col.startswith('hu_')]
    additional_features_test = additional_features_test.drop(columns=hu_cols_test, errors='ignore')
    additional_features_test_with_hu = additional_features_test.merge(hu_moments_test_df, on='image_path', how='left')
    additional_features_test_with_hu.to_csv('testFeatures/additional_features_with_hu.csv', index=False)
    return

def process_pipeline(hu: bool, prefix: str, plot_dir: str):
    # Select files
    add_feat = 'trainFeatures/additional_features_with_hu.csv' if hu else 'trainFeatures/additional_features.csv'
    add_feat_test = 'testFeatures/additional_features_with_hu.csv' if hu else 'testFeatures/additional_features.csv'
    summary_dir = f'summaries_{prefix}'
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    merged_df = merge_datasets(
        'trainFeatures/train_metadata.csv',
        'trainFeatures/hog_pca.csv',
        'trainFeatures/color_histogram.csv',
        add_feat
    )
    merged_df.to_csv(f'merged_dataset_{prefix}.csv', index=False)
    plot_class_distribution(merged_df, output_path=f'{summary_dir}/class_distribution.png')
    pca_summary, variance_df, top_features_loading, loading_df, pca_model, scaler_model = PCA_transform(merged_df, variance_exp=0.95)
    pca_summary.to_csv(f'{summary_dir}/pca_summary_features_{prefix}.csv', index=False)
    variance_df.to_csv(f'{summary_dir}/pca_summary_pc_{prefix}.csv', index=False)
    plot_pca_loadings_heatmap(loading_df, f"{summary_dir}/loadings_heatmap_{prefix}.png")
    processed_df = merged_df[['image_path', 'ClassId'] + top_features_loading]
    processed_df.to_csv(f'{summary_dir}/processed_dataset_{prefix}.csv', index=False)
    X = merged_df.drop(columns=['image_path', 'ClassId', 'id'], errors='ignore')
    X = X.loc[:, loading_df.columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    selected_pcs = loading_df.index.tolist()
    engineered_features_df = generate_pca_weighted_features(
        dataframe=X_scaled_df,
        loadings_df=loading_df,
        selected_pcs=selected_pcs,
        loading_threshold=0.1
    )
    engineered_features_df = pd.concat([merged_df[['image_path', 'ClassId']], engineered_features_df], axis=1)
    engineered_features_df.to_csv(f'{summary_dir}/engineered_df_all_{prefix}.csv', index=False)
    # Test set
    preprocess_test_dataset(
        'testFeatures/test_metadata.csv',
        'testFeatures/hog_pca.csv',
        'testFeatures/color_histogram.csv',
        add_feat_test,
        pca_model, loading_df, None, scaler_model, f'{summary_dir}/test_engineered_df_{prefix}.csv'
    )
    return

# ------------------- Main Script -------------------

save_hu_features()
# KNN: no Hu moments
process_pipeline(hu=False, prefix='no_hu', plot_dir='knn')
# SVM: with Hu moments
process_pipeline(hu=True, prefix='with_hu', plot_dir='svm')