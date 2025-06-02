import os

# --- DIRECTORIES ---
# Or os.getcwd() if running interactively
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")

# --- FILE PATHS ---
CSV_FILE_PATH = os.path.join(DATA_DIR, "dummy_exam_scores.csv")

# --- COLUMN NAMES (using constants for easier refactoring if names change) ---
COL_ID = "受験番号"
COL_LISTEN = "Listeningスコア"
COL_READ = "Readingスコア"
COL_TOTAL = "Totalスコア"
COL_GRADE = "成績"
COL_CLASS_CODE = "クラス(授業コード別)"
COL_PREDICTION = "prediction"

# --- KMEANS PARAMETERS ---
KMEANS_INIT = "random"
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 1
KMEANS_MAX_CLUSTERS_ELBOW = 10  # Max k for elbow method
KMEANS_CHOSEN_K = 10  # Default chosen k, can be overridden

# --- PLOTTING ---
DEFAULT_FIG_SIZE = (10, 6)
DENSITY_PLOT_FIG_SIZE = (7, 7)
DENSITY_PLOT_BW_METHOD = 0.8
DENSITY_PLOT_LINEWIDTH = 2

# --- OTHER ---
GRADE_CODES = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'F': 1, 'W': 0}
HISTOGRAM_SCORE_COLS = [COL_TOTAL, COL_LISTEN, COL_READ]

CASE_STUDIES = [
    (230, 40, "Case 1: Good Listening, Bad Reading"),
    (40, 230, "Case 2: Bad Listening, Good Reading"),
    (230, 230, "Case 3: Good Listening, Good Reading"),
    (120, 100, "Case 4: Average Listening, Average Reading")
]
