import pandas as pd
import numpy as np
from scipy import stats
import os
from config import GRADE_CODES


def map_grades_to_numeric(df, grade_col):
    """Maps categorical grades to numeric codes."""
    if grade_col in df.columns:
        df[grade_col] = df[grade_col].map(GRADE_CODES)
        print(f"Mapped '{grade_col}' to numeric codes.")
    else:
        print(f"Warning: Column '{grade_col}' not found for mapping.")
    return df


def pivot_and_save_data(df, score_key, group_by_col, output_dir, file_prefix_key, suffix):
    """Pivots data by group_by_col for score_key and saves to CSV."""
    if score_key not in df.columns or group_by_col not in df.columns:
        print(
            f"Warning: Required columns ('{score_key}' or '{group_by_col}') not found for pivot '{file_prefix_key} {suffix}'.")
        return None, None

    # Ensure score_key is numeric for pivoting
    df_copy = df.copy()
    df_copy[score_key] = pd.to_numeric(df_copy[score_key], errors='coerce')
    # Do not drop NaNs here, pivot handles it. Or fill with a specific value if needed.

    if df_copy.empty or df_copy[group_by_col].nunique() == 0:
        print(
            f"Warning: No data to pivot for '{score_key}' by '{group_by_col}' for '{file_prefix_key} {suffix}'.")
        return None, None

    pivoted_data = df_copy.pivot(columns=group_by_col, values=score_key)

    if pivoted_data.empty:
        print(
            f"Warning: Pivoted data is empty for '{score_key}' by '{group_by_col}'. Skipping CSV save.")
        return pivoted_data, None

    csv_filename = f'{file_prefix_key}_{suffix.lower().replace(" ", "_")}.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    pivoted_data.to_csv(csv_path, index=False)
    print(f"Saved pivoted data to {csv_path}")
    return pivoted_data, csv_path


def _prepare_data_for_ttest(pivoted_df):
    """Helper to prepare list of arrays from pivoted dataframe for t-tests."""
    data_list = []
    if pivoted_df is None or pivoted_df.empty:
        return data_list
    for col in pivoted_df.columns:
        # temp_series = pivoted_df[col].copy()
        # temp_series[np.isnan(temp_series)] = 0 # Original script filled NaNs with 0
        # data_list.append(temp_series[temp_series != 0].values)
        # Alternative: work with NaNs and let ttest_ind handle them with nan_policy='omit'
        data_list.append(pivoted_df[col].dropna().values)
    return data_list


def perform_ttests_on_pivoted_data(pivoted_data, analysis_label):
    """Performs t-tests between all pairs of groups in the pivoted data."""
    if pivoted_data is None or pivoted_data.empty or pivoted_data.shape[1] < 2:
        print(
            f"Skipping t-tests for {analysis_label}: Not enough groups or data in pivoted table.")
        return None, np.nan

    data_list = _prepare_data_for_ttest(pivoted_data)
    num_groups = len(data_list)
    if num_groups < 2:
        print(
            f"Skipping t-tests for {analysis_label}: Less than 2 groups after preparation.")
        return None, np.nan

    ptest_matrix = np.full((num_groups, num_groups),
                           np.nan)

    for i in range(num_groups):
        for j in range(num_groups):
            if i == j:
                # Diagonal is comparison with self, p-value is 1 or undefined.
                continue
            # Ensure there are at least 2 observations in each group for t-test
            if len(data_list[i]) < 2 or len(data_list[j]) < 2:
                ptest_matrix[i, j] = np.nan  # Not enough data for a t-test
                continue
            try:
                t_stat, p_value = stats.ttest_ind(
                    data_list[i], data_list[j], nan_policy='omit', equal_var=False)  # Welch's t-test
                ptest_matrix[i, j] = p_value
            except Exception as e:
                print(
                    f"Error during t-test between group {i} and {j} for {analysis_label}: {e}")
                ptest_matrix[i, j] = np.nan

    print(f"\nP-value matrix for {analysis_label}:\n", ptest_matrix)
    # Calculate average p-value, excluding NaNs and diagonal (which is already NaN)
    # Also exclude where ptest_matrix is 0 if that's a placeholder used. Here it's NaN.
    valid_p_values = ptest_matrix[~np.isnan(ptest_matrix)]
    average_p_value = np.nanmean(valid_p_values) if len(
        valid_p_values) > 0 else np.nan
    print(f"Average P-value ({analysis_label}): {average_p_value:.4f}")
    return ptest_matrix, average_p_value


def final_ttest_memo(pivoted_data_before_kmeans):
    """Performs the final 'Test and Memo' t-test from the original script."""
    if pivoted_data_before_kmeans is None or pivoted_data_before_kmeans.empty or pivoted_data_before_kmeans.shape[1] < 2:
        print("\nSkipping 'Test and Memo' t-test: Not enough data/groups in 'before KMeans' data.")
        return

    data_list = _prepare_data_for_ttest(pivoted_data_before_kmeans)

    if len(data_list) >= 2 and len(data_list[0]) > 1 and len(data_list[1]) > 1:
        print("\n--- Test and Memo: T-test between first two Reading score groups (before KMeans) ---")
        data1_test_memo = data_list[0]
        data2_test_memo = data_list[1]
        try:
            t_stat_memo, p_value_memo = stats.ttest_ind(
                data1_test_memo, data2_test_memo, nan_policy='omit', equal_var=False)
            print('Statistics=%.3f, p=%.3f' % (t_stat_memo, p_value_memo))
            alpha = 0.05
            if p_value_memo > alpha:
                print('Same distribution (fail to reject H0)')
            else:
                print('Different distribution (reject H0)')
        except Exception as e:
            print(f"Error during final 'Test and Memo' t-test: {e}")
    else:
        print("\nNot enough data/groups for the final 'Test and Memo' t-test.")


def process_score_analysis(df_full, score_key, class_col, prediction_col, output_dir, save_plots=False, save_csv=False):
    """
    Handles density plots, CSV saving, and t-tests for a specific score type (Listening/Reading).
    """
    print(f"\n--- Analysis for {score_key} ---")
    if score_key not in df_full.columns or class_col not in df_full.columns:
        print(
            f"Skipping {score_key} analysis: required columns ('{score_key}', '{class_col}') missing.")
        return None, None  # Return Nones for pivoted data if setup fails

    # --- Before KMeans Analysis (by Class) ---
    pivoted_before, _ = pivot_and_save_data(
        df_full, score_key, class_col, output_dir, f"before_k-mean_{score_key.replace('スコア','')}", "by_class")
    plot_score_density(df_full, score_key, class_col,
                       f"Before KMeans - by {class_col}", output_dir, save_plots=save_plots)
    ptest_matrix_before, avg_p_before = perform_ttests_on_pivoted_data(
        pivoted_before, f"{score_key} (Before KMeans - by {class_col})")

    # --- After KMeans Analysis (by Predicted Cluster) ---
    pivoted_after = None
    if prediction_col in df_full.columns:
        pivoted_after, _ = pivot_and_save_data(
            df_full, score_key, prediction_col, output_dir, f"after_k-mean_{score_key.replace('スコア','')}", "by_cluster")
        plot_score_density(df_full, score_key, prediction_col,
                           "After KMeans - by Cluster", output_dir, save_plots=save_plots)
        ptest_matrix_after, avg_p_after = perform_ttests_on_pivoted_data(
            pivoted_after, f"{score_key} (After KMeans - by Cluster)")
    else:
        print(
            f"Skipping 'after KMeans' density plot and t-tests for {score_key} as '{prediction_col}' column is missing.")

    # Return pivoted data for potential further use (like the final t-test memo)
    return pivoted_before, pivoted_after
