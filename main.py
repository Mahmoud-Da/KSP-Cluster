import config
from data_loader import load_data, initial_preprocess, ensure_output_dir
from plotting_utils import (generate_histograms, plot_scatter_original,
                            plot_clustered_data_with_centroids)
from clustering_utils import (prepare_data_for_kmeans, run_elbow_method,
                              fit_kmeans_model, predict_case_studies, add_predictions_to_df)
from analysis_utils import (map_grades_to_numeric,
                            process_score_analysis, final_ttest_memo)


def main():
    # --- Setup ---
    ensure_output_dir(config.OUTPUT_DIR)

    # --- Data Loading and Initial Preprocessing ---
    raw_dataframe = load_data(config.CSV_FILE_PATH)
    df_cleaned = initial_preprocess(raw_dataframe)

    if df_cleaned.empty:
        print("Exiting: DataFrame is empty after initial processing.")
        return

    # --- Histograms ---
    # Convert relevant columns to numeric for histograms if not already
    # This step should ideally ensure they are numeric before passing.
    # data_loader.initial_preprocess already attempts numeric conversion.
    generate_histograms(df_cleaned.copy(
    ), config.HISTOGRAM_SCORE_COLS, config.OUTPUT_DIR, save_plots=True)

    # --- KMeans Clustering ---
    print("\n--- KMeans Clustering ---")
    # Prepare data for KMeans
    df_for_kmeans_original, scaler, scaled_data_for_kmeans = prepare_data_for_kmeans(
        df_cleaned.copy(), config.COL_LISTEN, config.COL_READ
    )

    kmeans_model = None
    kmeans_labels = None
    kmeans_centroids_scaled = None
    df_with_predictions = df_cleaned.copy()  # Start with the cleaned df

    if scaled_data_for_kmeans is not None and scaler is not None and df_for_kmeans_original is not None:
        # Elbow Method
        run_elbow_method(scaled_data_for_kmeans,
                         config.OUTPUT_DIR, save_plots=True)
        # User might inspect elbow plot and decide k. For automation, use config.KMEANS_CHOSEN_K
        chosen_k = config.KMEANS_CHOSEN_K
        print(f"Proceeding with k = {chosen_k} for KMeans.")

        # Fit KMeans Model
        kmeans_model, kmeans_labels, kmeans_centroids_scaled = fit_kmeans_model(
            scaled_data_for_kmeans, chosen_k)

        if kmeans_model and kmeans_labels is not None and kmeans_centroids_scaled is not None:
            # Inverse transform centroids and data for plotting
            original_centroids_kmeans = scaler.inverse_transform(
                kmeans_centroids_scaled)
            # df_for_kmeans_original now holds the unscaled data corresponding to scaled_data_for_kmeans

            # Plot original data (unscaled) colored by cluster
            plot_scatter_original(
                df_for_kmeans_original.values,  # Pass as numpy array
                kmeans_labels,
                config.COL_LISTEN,
                config.COL_READ,
                config.OUTPUT_DIR,
                save_plots=True
            )

            # Plot clustered data with centroids
            plot_clustered_data_with_centroids(
                df_for_kmeans_original.values,  # Pass as numpy array
                kmeans_labels,
                original_centroids_kmeans,
                config.COL_LISTEN,
                config.COL_READ,
                config.OUTPUT_DIR,
                save_plots=True
            )

            # Case Studies
            predict_case_studies(scaler, kmeans_model, config.CASE_STUDIES)

            # Add predictions to the full cleaned dataframe
            df_with_predictions = add_predictions_to_df(
                df_cleaned.copy(), scaler, kmeans_model,
                config.COL_LISTEN, config.COL_READ, config.COL_PREDICTION
            )
            print("\nDataFrame with predictions head:")
            print(df_with_predictions.head())
        else:
            print(
                "KMeans model fitting failed or produced no labels/centroids. Skipping subsequent KMeans steps.")
    else:
        print("Data preparation for KMeans failed. Skipping KMeans clustering.")

    # --- Further Analysis ---
    print("\n--- Further Analysis ---")
    # Map '成績' column if it exists
    df_for_further_analysis = map_grades_to_numeric(
        df_with_predictions.copy(), config.COL_GRADE)

    # Analyze Listening Scores
    _, _ = process_score_analysis(
        df_for_further_analysis,
        config.COL_LISTEN,
        config.COL_CLASS_CODE,
        config.COL_PREDICTION,
        config.OUTPUT_DIR,
        save_plots=True,
        save_csv=True
    )

    # Analyze Reading Scores
    pivoted_reading_before, _ = process_score_analysis(
        df_for_further_analysis,
        config.COL_READ,
        config.COL_CLASS_CODE,
        config.COL_PREDICTION,
        config.OUTPUT_DIR,
        save_plots=True,
        save_csv=True
    )

    # Final "Test and Memo" t-test (uses reading scores before KMeans)
    if pivoted_reading_before is not None:
        final_ttest_memo(pivoted_reading_before)
    else:
        print("Skipping final 'Test and Memo' t-test as 'pivoted_reading_before' data is not available.")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()
