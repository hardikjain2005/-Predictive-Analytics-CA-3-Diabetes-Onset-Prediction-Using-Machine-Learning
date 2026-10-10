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

# The project's own GitHub repository: the notebook reads the dataset from here
# when it runs outside the project folder (e.g. in Google Colab).
GITHUB_REPO = ("hardikjain2005/"
               "-Predictive-Analytics-CA-3-Diabetes-Onset-Prediction-Using-Machine-Learning")
GITHUB_DATA_URL = (f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/"
                   f"data/{DATA_FILENAME}")
# Original header-less source of the dataset
ORIGINAL_SOURCE_URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
                       "pima-indians-diabetes.data.csv")


def _read_url(url):
    """Return the text at url.

    Uses certifi's certificate bundle when available, because some Python
    installations (e.g. python.org builds on macOS) ship without root
    certificates and would otherwise fail SSL verification.
    """
    import ssl
    import urllib.request

    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(url, context=context, timeout=30) as response:
        return response.read().decode("utf-8")


def download_data(dest, url=GITHUB_DATA_URL):
    """Download the dataset from url and save it (with headers) at dest."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    load_data(url).to_csv(dest, index=False)
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

    path can be a local file or an http(s) URL (e.g. GITHUB_DATA_URL). Works
    both for the headered CSV in data/ and for the original header-less file
    from the jbrownlee/Datasets repository.
    """
    import io

    if path is None:
        path = find_data_file()
    if str(path).startswith(("http://", "https://")):
        text = _read_url(str(path))
        read = lambda **kw: pd.read_csv(io.StringIO(text), **kw)
    else:
        read = lambda **kw: pd.read_csv(path, **kw)
    df = read()
    if list(df.columns) != COLUMNS:
        df = read(header=None, names=COLUMNS)
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
