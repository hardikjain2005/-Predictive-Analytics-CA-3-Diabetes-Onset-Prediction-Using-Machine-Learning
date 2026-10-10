"""Data loading, cleaning and preprocessing helpers for the diabetes project.

All functions that *learn* something from data (imputer medians, scaler
means/standard deviations) are returned as unfitted scikit-learn objects so
that they are fitted on the training split only. This keeps the test set
completely unseen and avoids data leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"
COLUMNS = FEATURES + [TARGET]

# A value of 0 is physiologically impossible for these measurements,
# so in this dataset 0 is a placeholder for "not recorded".
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

DATA_FILENAME = "pima-indians-diabetes.csv"
SOURCE_URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
              "pima-indians-diabetes.data.csv")


def download_data(dest, url=SOURCE_URL):
    """Download the original header-less CSV and save it with column headers.

    Uses certifi's certificate bundle when available, because some Python
    installations (e.g. python.org builds on macOS) ship without root
    certificates and would otherwise fail SSL verification.
    """
    import io
    import ssl
    import urllib.request

    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(url, context=context, timeout=30) as response:
        text = response.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(text), header=None, names=COLUMNS)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def find_data_file(filename=DATA_FILENAME):
    """Locate the dataset whether the code runs from the project root,
    from notebooks/ or from src/."""
    here = Path.cwd()
    candidates = [
        here / "data" / filename,
        here.parent / "data" / filename,
        Path(__file__).resolve().parent.parent / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {filename}. Place the Pima Indians Diabetes CSV in the "
        "project's data/ folder (see README.md, section 'Dataset')."
    )


def load_data(path=None):
    """Load the Pima Indians Diabetes dataset as a DataFrame.

    Works both for the headered CSV shipped in data/ and for the original
    header-less file from the jbrownlee/Datasets repository.
    """
    path = Path(path) if path is not None else find_data_file()
    df = pd.read_csv(path)
    if list(df.columns) != COLUMNS:
        df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def count_invalid_zeros(df, columns=ZERO_AS_MISSING):
    """Table of how many zeros appear in columns where zero is invalid."""
    counts = (df[columns] == 0).sum()
    return pd.DataFrame(
        {
            "Invalid zeros": counts,
            "Percentage (%)": (counts / len(df) * 100).round(2),
        }
    )


def replace_invalid_zeros(df, columns=ZERO_AS_MISSING):
    """Return a copy of df where invalid zeros are replaced with NaN.

    This is a row-wise rule that does not learn anything from the data,
    so applying it before the train/test split cannot leak information.
    """
    df = df.copy()
    df[columns] = df[columns].replace(0, np.nan)
    return df


def split_data(df, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train/test split that keeps the class ratio in both sets."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def build_preprocessor(scale=True):
    """Median imputation (+ optional standardisation) as an unfitted Pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
