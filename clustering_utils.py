from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from config import KMEANS_INIT, KMEANS_N_INIT, KMEANS_RANDOM_STATE, KMEANS_MAX_CLUSTERS_ELBOW, COL_LISTEN, COL_READ
from plotting_utils import plot_elbow_method


def prepare_data_for_kmeans(df, listen_col, read_col):
    """Prepares and scales data for KMeans, returns scaler and scaled_df."""
    if listen_col not in df.columns or read_col not in df.columns:
        print(
            f"Warning: '{listen_col}' or '{read_col}' not found. Cannot prepare data for KMeans.")
        return None, None, None

    # Ensure columns are numeric, fill NaNs with 0 (or another strategy)
    df_kmeans_subset = df[[listen_col, read_col]].copy()
    df_kmeans_subset[listen_col] = pd.to_numeric(
        df_kmeans_subset[listen_col], errors='coerce').fillna(0).astype(int)
    df_kmeans_subset[read_col] = pd.to_numeric(
        df_kmeans_subset[read_col], errors='coerce').fillna(0).astype(int)

    if df_kmeans_subset.empty:
        print("Warning: DataFrame for KMeans is empty after cleaning.")
        return None, None, None

    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df_kmeans_subset)
    return df_kmeans_subset, scaler, scaled_df


def run_elbow_method(scaled_df, output_dir, save_plots=False):
    """Runs the elbow method to find optimal k and plots it."""
    if scaled_df is None or len(scaled_df) == 0:
        print("Skipping Elbow method: no scaled data.")
        return

    print("\n--- KMeans Clustering: Elbow Method ---")
    sse = []
    # Adjust max_k if dataset is too small
    max_k = min(KMEANS_MAX_CLUSTERS_ELBOW, len(
        scaled_df) - 1 if len(scaled_df) > 1 else 1)
    if max_k < 1:
        print("Not enough samples for elbow method.")
        return

    for k in range(1, max_k + 1):
        # 'lloyd' algorithm is now default for KMeans.
        # Earlier versions used 'full' by default, which became 'lloyd'.
        # The 'auto' algorithm choice was deprecated. Explicitly using 'lloyd'.
        kmeans_elbow = KMeans(n_clusters=k, init=KMEANS_INIT, n_init=KMEANS_N_INIT,
                              random_state=KMEANS_RANDOM_STATE, algorithm='lloyd')
        kmeans_elbow.fit(scaled_df)
        sse.append(kmeans_elbow.inertia_)
    plot_elbow_method(sse, max_k, output_dir, save_plots=save_plots)


def fit_kmeans_model(scaled_df, chosen_k):
    """Fits the KMeans model with a chosen k."""
    if scaled_df is None or len(scaled_df) == 0:
        print("Skipping KMeans model fitting: no scaled data.")
        return None, None, None

    actual_k = chosen_k
    if chosen_k >= len(scaled_df) and len(scaled_df) > 0:
        print(
            f"Warning: n_clusters ({chosen_k}) must be less than n_samples ({len(scaled_df)}). Adjusting k.")
        actual_k = max(1, len(scaled_df) - 1) if len(scaled_df) > 1 else 1
        print(f"Adjusted k to {actual_k}")
    elif len(scaled_df) == 0:
        print("Cannot fit KMeans: 0 samples.")
        return None, None, None

    kmeans = KMeans(n_clusters=actual_k, init=KMEANS_INIT, n_init=KMEANS_N_INIT,
                    random_state=KMEANS_RANDOM_STATE, algorithm='lloyd')
    model = kmeans.fit(scaled_df)
    labels = model.predict(scaled_df)
    centroids = model.cluster_centers_
    print(f"KMeans model fitted with k={actual_k}.")
    return model, labels, centroids


def predict_case_studies(scaler, kmeans_model, case_studies_data):
    """Predicts clusters for predefined case studies."""
    if scaler is None or kmeans_model is None:
        print("Skipping case studies: scaler or KMeans model not available.")
        return

    print("\n--- Case Studies ---")
    for listen_score, read_score, desc in case_studies_data:
        try:
            # Ensure input is 2D array for transform
            scaled_student_score = scaler.transform(
                np.array([[listen_score, read_score]]))
            predicted_label = kmeans_model.predict(scaled_student_score)
            print(f"{desc} -> Predicted Cluster: {predicted_label[0]}")
        except Exception as e:
            print(f"Error predicting for case study '{desc}': {e}")


def add_predictions_to_df(df, scaler, kmeans_model, listen_col, read_col, pred_col_name):
    """Adds KMeans cluster predictions to the DataFrame."""
    if scaler is None or kmeans_model is None:
        print("Skipping adding predictions: scaler or KMeans model not available.")
        return df
    if listen_col not in df.columns or read_col not in df.columns:
        print(
            f"Warning: '{listen_col}' or '{read_col}' not in DataFrame for adding predictions.")
        return df

    df_copy = df.copy()
    # Ensure columns are numeric for scaling, fill NaNs with 0 or other strategy
    df_copy[listen_col] = pd.to_numeric(
        df_copy[listen_col], errors='coerce').fillna(0).astype(int)
    df_copy[read_col] = pd.to_numeric(
        df_copy[read_col], errors='coerce').fillna(0).astype(int)

    features_for_prediction = df_copy[[listen_col, read_col]]
    if features_for_prediction.empty:
        print("No data to add predictions to.")
        return df_copy

    scaled_data_full = scaler.transform(features_for_prediction)
    df_copy[pred_col_name] = kmeans_model.predict(scaled_data_full)
    print(f"\nAdded '{pred_col_name}' column to DataFrame.")
    return df_copy
