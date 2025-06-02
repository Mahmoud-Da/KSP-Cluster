import pandas as pd
import os
import sys
from config import COL_ID, COL_LISTEN, COL_READ, COL_TOTAL


def load_data(csv_file_path):
    """Loads data from the CSV file."""
    print("--- Initial Data Loading and Preprocessing ---")
    try:
        dataframe = pd.read_csv(csv_file_path, header=None,
                                names=[COL_ID, COL_LISTEN, COL_READ, COL_TOTAL])
    except FileNotFoundError:
        print(
            f"Error: {csv_file_path} not found. Please ensure it's in the correct directory.")
        sys.exit(1)
    return dataframe


def initial_preprocess(dataframe):
    """Performs initial preprocessing steps on the loaded dataframe."""
    if dataframe.empty:
        print("Error: DataFrame is empty after loading.")
        sys.exit(1)

    # Set columns from the first row if it looks like headers
    # A more robust check might be needed if first row isn't always headers
    if all(isinstance(x, str) for x in dataframe.iloc[0]):
        print("Setting column names from first row.")
        # Use predefined constants if they match, otherwise use values from CSV
        # This part is a bit tricky because original script renames to Japanese
        # For now, let's assume the first row has the Japanese names
        dataframe.columns = dataframe.iloc[0].values.tolist()
        dataframe = dataframe.drop(
            index=dataframe.index[0], axis=0).reset_index(drop=True)
    else:
        # If first row doesn't look like headers, assign default names used in config
        print("Assigning default column names.")
        # Ensure we have enough columns
        if dataframe.shape[1] >= 4:
            dataframe.columns = [COL_ID, COL_LISTEN, COL_READ, COL_TOTAL] + \
                [f"extra_col_{i}" for i in range(dataframe.shape[1] - 4)]
        else:  # Or handle error if not enough columns
            print(
                f"Warning: DataFrame has only {dataframe.shape[1]} columns, expected at least 4.")
            # Fallback to generic naming if fewer than 4 columns
            dataframe.columns = [
                f"col_{i+1}" for i in range(dataframe.shape[1])]

    # Clean data: Drop rows where 'Totalスコア' is NaN or '欠席'
    if COL_TOTAL in dataframe.columns:
        dataframe = dataframe.dropna(subset=[COL_TOTAL])
        dataframe = dataframe[dataframe[COL_TOTAL] != '欠席']
    else:
        print(f"Warning: Column '{COL_TOTAL}' not found for cleaning.")

    # Convert score columns to numeric early, coercing errors
    # This makes subsequent operations cleaner
    score_cols_to_convert = [col for col in [
        COL_TOTAL, COL_LISTEN, COL_READ] if col in dataframe.columns]
    for col in score_cols_to_convert:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    # Optionally drop rows where essential scores became NaN after conversion
    # dataframe = dataframe.dropna(subset=[COL_LISTEN, COL_READ]) # if they are crucial for all steps

    dataframe_full_cleaned = dataframe.copy()
    print(f"Shape of dataframe_full_cleaned: {dataframe_full_cleaned.shape}")
    return dataframe_full_cleaned


def ensure_output_dir(output_dir):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
