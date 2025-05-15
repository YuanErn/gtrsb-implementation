import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import warnings
import os

def merge_datasets(metadata_path, hog_path, color_hist_path, additional_feat_path):
    """
    Merges all feature datasets into a single DataFrame, removing duplicate columns.
    """
    import pandas as pd

    # Load data
    train_metadata = pd.read_csv(metadata_path)
    hog_pca = pd.read_csv(hog_path)
    color_histogram = pd.read_csv(color_hist_path)
    additional_features = pd.read_csv(additional_feat_path)

    # Merge on 'image_path'
    merged = train_metadata \
        .merge(hog_pca, on='image_path', how='inner') \
        .merge(color_histogram, on='image_path', how='inner') \
        .merge(additional_features, on='image_path', how='inner')

    # Drop duplicate columns (same name and values)
    merged_df = merged.loc[:, ~merged.columns.duplicated()]

    return merged_df

def plot_class_distribution(df, output_path='data_analysis/class_distribution.png'):
    """
    Plots and saves a histogram of class label distribution.
    """
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

def PCA_transform(dataframe, variance_exp):
    # Extract feature matrix, and remove the id and label columns, dropping any NaN or infinite values
    X = dataframe.drop(columns=['image_path', 'ClassId', 'id'], errors='ignore')
    mask_finite = np.isfinite(X).all(axis=0)
    X = X.loc[:, mask_finite]
    feature_names = list(X.columns)

    # Drop zero-variance columns
    sel = VarianceThreshold(threshold=0.0)
    sel.fit(X)
    keep_mask = sel.get_support()
    X = X.loc[:, keep_mask]
    feature_names = [f for f, keep in zip(feature_names, keep_mask) if keep]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA, we compute the PCA with all components first to determine the number of components needed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pca = PCA(svd_solver='auto')
        pca.fit(X_scaled)

    cum_var = pca.explained_variance_ratio_.cumsum()
    n_components = (cum_var < variance_exp).sum() + 1

    # Re-fit with chosen number of components
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pca = PCA(svd_solver = 'auto', n_components=n_components)
        pca.fit_transform(X_scaled)

    # Calculate variance contribution for each feature
    loadings = pca.components_  # shape (n_components, n_features)
    var_ratios = pca.explained_variance_ratio_  # shape (n_components,)
    feature_contributions = np.sum((loadings ** 2) * var_ratios[:, np.newaxis], axis=0)
    feature_contributions /= feature_contributions.sum()

    # Ranked DataFrame
    ranked_df = pd.DataFrame({
        'Feature': feature_names,
        'Variance Contribution': feature_contributions
    }).sort_values(by='Variance Contribution', ascending=False).reset_index(drop=True)

    # Determine features contributing to given variance_exp threshold
    ranked_df['Cumulative'] = ranked_df['Variance Contribution'].cumsum()
    selected_features = ranked_df[ranked_df['Cumulative'] <= variance_exp]['Feature'].tolist()
    
    # Add the next feature if threshold isn't exactly hit
    if ranked_df['Cumulative'].iloc[len(selected_features)] < variance_exp:
        selected_features.append(ranked_df['Feature'].iloc[len(selected_features)])

    # Create loadings DataFrame for heatmap
    loadings_df = pd.DataFrame(
        loadings,
        columns=feature_names,
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Create a DataFrame
    variance_df = pd.DataFrame({
        'PC': [f'PC{i+1}' for i in range(len(explained_var))],
        'ExplainedVariance': explained_var,
        'CumulativeVariance': cumulative_var
    })


    return ranked_df, variance_df, selected_features, loadings_df, pca, scaler

def plot_pca_loadings_heatmap(loadings_df, save_path):
    """
    Plots a heatmap of PCA component loadings from a loadings DataFrame.
    """
    # Dynamic figure sizing
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

def generate_pca_weighted_features(dataframe, loadings_df, selected_pcs, loading_threshold=0.1):
    """
    Create new features as weighted sums of original features based on PCA loadings.
    This is useful for dimensionality reduction and feature engineering from pca heatmap.
    The heatmap is visually inspected to select the most relevant PCs which is passed as a list.
    """
    if selected_pcs is None:
        selected_pcs = loadings_df.index.tolist()  # Use all PCs by default

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

        # Ensure all selected features exist in the input dataframe
        available_features = [f for f in selected_features.index if f in dataframe.columns]
        weights = selected_features[available_features]

        if weights.empty:
            print(f"No matching features found in dataframe for {pc}")
            continue

        engineered_features[pc + '_weighted'] = dataframe[weights.index] @ weights.values

    return pd.DataFrame(engineered_features)

def preprocess_test_dataset(test_metadata_path, hog_path, color_hist_path, additional_feat_path, pca_model, loading_df, selected_pcs, scaler_model):
    """
    Preprocesses the test dataset by merging features and saving the result.
    """
    merged_df = merge_datasets(
    test_metadata_path,
    hog_path,
    color_hist_path,
    additional_feat_path
    )

    # Drop duplicate columns (same name and values)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    X_test = merged_df.drop(columns=['image_path'], errors='ignore')
    X_test = X_test.loc[:, loading_df.columns]

    # Scale using training scaler
    X_test_scaled = scaler_model.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Now apply feature engineering
    engineered_features_df = generate_pca_weighted_features(
        dataframe=X_test_scaled_df,
        loadings_df=loading_df,
        selected_pcs=selected_pcs,
        loading_threshold=0.1
    )
    # Add to final DataFrame
    engineered_features_df = pd.concat([merged_df[['image_path']], engineered_features_df], axis=1)
    engineered_features_df.to_csv('test_engineered_df.csv', index=False)
    return 


# Merging and saving the datasets
merged_df = merge_datasets(
    'trainFeatures/train_metadata.csv',
    'trainFeatures/hog_pca.csv',
    'trainFeatures/color_histogram.csv',
    'trainFeatures/additional_features.csv'
)
merged_df.to_csv('merged_dataset.csv', index=False)

# Plotting histogram for class distribution
plot_class_distribution(merged_df)

# Performing PCA and saving the results, plotting a heatmap of loadings
pca_summary, variance_df, top_features_loading, loading_df, pca_model, scaler_model = PCA_transform(merged_df, variance_exp=0.95)
pca_summary.to_csv('pca_summary_features.csv', index=False)
variance_df.to_csv('pca_summary_pc.csv', index=False)
plot_pca_loadings_heatmap(loading_df, "knn/loadings_heatmap.png")

# Making a new DataFrame with PCA components (selected top contributing features)
processed_df = merged_df[['image_path', 'ClassId'] + top_features_loading]
processed_df.to_csv('processed_dataset.csv', index=False)

# ------------------------
# Feature Engineering from PCA Loadings
# ------------------------

X = merged_df.drop(columns=['image_path', 'ClassId', 'id'], errors='ignore')
X = X.loc[:, loading_df.columns] 

# Standardize using the same logic as PCA_transform
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Generate engineered features using selected PCs
selected_pcs = ['PC16','PC20', 'PC21', 'PC23', 'PC24', 'PC25', 'PC26', 'PC27', 'PC28', 'PC29', 'PC30', 'PC31', 'PC32', 'PC43', 'PC44','PC46', 'PC49']  
engineered_features_df = generate_pca_weighted_features(
    dataframe=X_scaled_df,
    loadings_df=loading_df,
    selected_pcs=selected_pcs,
    loading_threshold=0.1 
)

# Add to final DataFrame for selecting pcs
engineered_features_df = pd.concat([merged_df[['image_path', 'ClassId']], engineered_features_df], axis=1)
engineered_features_df.to_csv('engineered_df_selected.csv', index=False)

# Making a new DataFrame with PCA components, without selection
engineered_features_df = generate_pca_weighted_features(
    dataframe=X_scaled_df,
    loadings_df=loading_df,
    selected_pcs=None,
    loading_threshold=0.1 
)

engineered_features_df = pd.concat([merged_df[['image_path', 'ClassId']], engineered_features_df], axis=1)
engineered_features_df.to_csv('engineered_df_all.csv', index=False)

# Load the test dataset that has been preprocessed
test_metadata_path = 'testFeatures/test_metadata.csv'
hog_path = 'testFeatures/hog_pca.csv'
color_hist_path = 'testFeatures/color_histogram.csv'
additional_feat_path = 'testFeatures/additional_features.csv'
selected_pcs = loading_df.index.tolist()
X_test_pca = preprocess_test_dataset(test_metadata_path, hog_path, color_hist_path, additional_feat_path, pca_model, loading_df, None, scaler_model)