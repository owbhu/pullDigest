from dotenv import load_dotenv
load_dotenv()
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_LIST = [
    "django/django",
    "numpy/numpy",
]

DB_PATH = "data/prs.db"
HF_MODEL = "google/flan-t5-small"
HF_TOKEN = os.getenv("HF_TOKEN")
MANUAL_LABELS_CSV = "data/manual_labels.csv"
