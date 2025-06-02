import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from config import DEFAULT_FIG_SIZE, DENSITY_PLOT_FIG_SIZE, DENSITY_PLOT_BW_METHOD, DENSITY_PLOT_LINEWIDTH

# Optional: Configure for Japanese characters if needed
# try:
#     plt.rcParams['font.family'] = 'IPAexGothic' # Or other Japanese font like 'MS Gothic', 'Meiryo'
# except RuntimeError:
#     print("Warning: Japanese font not found. Plots may not render Japanese characters correctly.")


def generate_histograms(df, score_columns, output_dir, save_plots=False):
    """Generates and optionally saves histograms for specified score columns."""
    print("\n--- Generating Histograms ---")
    for col_name in score_columns:
        if col_name in df.columns:
            try:
                # Ensure data is numeric and clean for histogram
                # Already converted in data_loader, but double check for NaNs
                temp_scores = pd.to_numeric(
                    df[col_name], errors='coerce').dropna()

                if not temp_scores.empty:
                    min_score = temp_scores.min()
                    max_score = temp_scores.max()

                    if max_score > min_score:
                        bin_width = (max_score - min_score) // 15
                        if bin_width == 0:
                            bin_width = 1
                        bins = np.arange(
                            min_score, max_score + bin_width, bin_width)
                        # Handle case where max_score is exactly at a bin edge
                        if bins[-1] < max_score:
                            bins = np.append(bins, bins[-1] + bin_width)

                        plt.figure(figsize=DEFAULT_FIG_SIZE)
                        # Use pd.cut for binning and then plot counts
                        pd.cut(temp_scores, bins, right=False).value_counts(
                        ).sort_index().plot(kind='bar')
                        plt.title(f'Histogram of {col_name}')
                        plt.xlabel(col_name)
                        plt.ylabel('Frequency')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        if save_plots:
                            filename = f'histogram_{col_name.replace("スコア","").replace("(","").replace(")","").replace("別","")}.png'
                            plt.savefig(os.path.join(output_dir, filename))
                        plt.show()
                    else:
                        print(
                            f"Skipping histogram for {col_name}: Not enough data range (min={min_score}, max={max_score}).")
                else:
                    print(
                        f"Skipping histogram for {col_name}: No valid numeric data after cleaning.")
            except Exception as e:
                print(f"Error processing column {col_name} for histogram: {e}")
        else:
            print(f"Warning: Column '{col_name}' not found for histogram.")


def plot_elbow_method(sse_values, max_clusters, output_dir, save_plots=False):
    """Plots the SSE for the elbow method."""
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    plt.plot(range(1, max_clusters + 1), sse_values)
    plt.xticks(range(1, max_clusters + 1))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title("Elbow Method for Optimal k")
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'elbow_method.png'))
    plt.show()


def plot_scatter_original(original_data, labels, col_listen, col_read, output_dir, save_plots=False):
    """Plots scatter of original unscaled data with cluster labels."""
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    plt.scatter(original_data[:, 0], original_data[:, 1],
                c=labels, cmap='viridis', alpha=0.5)
    plt.title('Scatter plot of original data (unscaled) by Cluster')
    plt.xlabel(col_listen)
    plt.ylabel(col_read)
    if save_plots:
        plt.savefig(os.path.join(
            output_dir, 'scatter_original_unscaled_colored.png'))
    plt.show()


def plot_clustered_data_with_centroids(original_data, labels, centroids, col_listen, col_read, output_dir, save_plots=False):
    """Plots clustered data with centroids."""
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    unique_labels = np.unique(labels)
    for i in unique_labels:
        plt.scatter(original_data[labels == i, 0],
                    original_data[labels == i, 1], label=f'Cluster {i}')
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    color='black', marker='X', s=100, label='Centroids')
    plt.title('Clustered Data with Centroids')
    plt.xlabel(col_listen)
    plt.ylabel(col_read)
    plt.legend()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'clustered_data_centroids.png'))
    plt.show()


def plot_score_density(df, score_key, group_by_col, title_suffix, output_dir, plot_target_ax=None, save_plots=False):
    """Generates and optionally saves density plots for a score, grouped by a column."""
    if score_key not in df.columns or group_by_col not in df.columns:
        print(
            f"Warning: Required columns ('{score_key}' or '{group_by_col}') not in DataFrame for density plot '{title_suffix}'.")
        return None

    # NOTE: Ensure score_key is numeric for pivoting and plotting
    df_copy = df.copy()
    df_copy[score_key] = pd.to_numeric(df_copy[score_key], errors='coerce')
    # Drop rows where score is NaN after conversion
    df_copy = df_copy.dropna(subset=[score_key])

    if df_copy.empty or df_copy[group_by_col].nunique() == 0:
        print(
            f"Warning: No data to plot for '{score_key}' grouped by '{group_by_col}' for density plot '{title_suffix}'.")
        return None

    pivoted_data = df_copy.pivot(columns=group_by_col, values=score_key)

    if pivoted_data.empty:
        print(
            f"Warning: Pivoted data is empty for '{score_key}' grouped by '{group_by_col}'. Skipping density plot.")
        return pivoted_data

    # NOTE: Create a new figure if no axis is provided
    if plot_target_ax is None:
        fig, ax = plt.subplots(figsize=DENSITY_PLOT_FIG_SIZE)
    else:
        ax = plot_target_ax

    pivoted_data.plot.density(
        bw_method=DENSITY_PLOT_BW_METHOD,
        linewidth=DENSITY_PLOT_LINEWIDTH,
        ax=ax
    )
    ax.set_xlabel(score_key)
    ax.set_title(f'Density Plot of {score_key} ({title_suffix})')

    # NOTE: Only save if it's a new plot
    if save_plots and plot_target_ax is None:
        filename = f'density_{score_key.replace("スコア","")}_{title_suffix.lower().replace(" ", "_").replace("-","").replace("(","").replace(")","")}.png'
        plt.savefig(os.path.join(output_dir, filename))
    if plot_target_ax is None:
        plt.show()
    return pivoted_data
