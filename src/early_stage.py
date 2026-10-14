"""Dataset 2: Early Stage Diabetes Risk Prediction (UCI Machine Learning Repository, id 529).

520 questionnaire records from patients of Sylhet Diabetes Hospital (Bangladesh):
Age (numeric), Gender and 14 Yes/No symptom columns, and a Positive/Negative
class. Licensed CC BY 4.0, DOI 10.24432/C5VG8H.

Nothing here assumes the Pima column names. The target column and the feature
types are detected from the data itself, and every step that *learns* from data
(imputation, encoding, scaling) is returned as an unfitted object so it is
fitted on the training split only.
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing import GITHUB_REPO, _read_url

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:  # handled gracefully: the model is simply skipped
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42

ES_FILENAME = "early_stage_diabetes_risk_prediction.csv"
UCI_ZIP_URL = ("https://archive.ics.uci.edu/static/public/529/"
               "early+stage+diabetes+risk+prediction+dataset.zip")
UCI_CSV_NAME = "diabetes_data_upload.csv"
GITHUB_ES_URL = (f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/"
                 f"data/{ES_FILENAME}")

# Names commonly used for a target column, checked case-insensitively
TARGET_CANDIDATES = ("class", "outcome", "target", "label", "diagnosis", "diabetes")
POSITIVE_LABELS = {"positive", "yes", "1", "true", "diabetic", "diabetes"}
# Tokens that mean "not recorded" in survey-style CSV files
MISSING_TOKENS = {"", "?", "na", "n/a", "nan", "none", "null", "unknown", "-"}


# ------------------------------------------------------------------ loading
def download_early_stage(dest):
    """Download the dataset from UCI (zip) and save it as a CSV at dest.
    Falls back to the copy in this project's GitHub repository."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import ssl
        import urllib.request
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with urllib.request.urlopen(UCI_ZIP_URL, context=context, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        with archive.open(UCI_CSV_NAME) as fh:
            df = pd.read_csv(fh)
        source = UCI_ZIP_URL
    except Exception:
        df = pd.read_csv(io.StringIO(_read_url(GITHUB_ES_URL)))
        source = GITHUB_ES_URL
    df.to_csv(dest, index=False)
    return dest, source


def load_early_stage(path):
    """Read the dataset from a local path or an http(s) URL."""
    if str(path).startswith(("http://", "https://")):
        return pd.read_csv(io.StringIO(_read_url(str(path))))
    return pd.read_csv(path)


# ------------------------------------------------------------------ understanding / cleaning
def find_target_column(df):
    """Identify the target column automatically.

    1. a column whose name is a common target name (class, outcome, ...);
    2. otherwise the last column with exactly two distinct values.
    """
    lower = {c.strip().lower(): c for c in df.columns}
    for name in TARGET_CANDIDATES:
        if name in lower:
            return lower[name]
    binary = [c for c in df.columns if df[c].dropna().nunique() == 2]
    if not binary:
        raise ValueError("No binary target column could be identified.")
    return binary[-1]


def encode_target(series):
    """Map the target to 1 (diabetes / positive) and 0 (negative).

    Returns the encoded Series and the mapping used, so it can be shown."""
    values = series.dropna().astype(str).str.strip().unique()
    mapping = {v: int(v.lower() in POSITIVE_LABELS) for v in values}
    if sorted(mapping.values()) != [0, 1]:
        raise ValueError(f"Could not tell which target value is positive: {list(values)}")
    return series.astype(str).str.strip().map(mapping).astype(int), mapping


def clean_categories(df):
    """Row-level text clean-up that learns nothing from the data (so it cannot
    leak information): trim whitespace, unify Yes/No capitalisation and turn
    'unknown'-style tokens into NaN, to be imputed later inside the pipeline."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string").str.strip()
            s = s.mask(s.str.lower().isin(MISSING_TOKENS))
            s = s.str.capitalize() if set(s.dropna().str.lower().unique()) <= {"yes", "no"} else s
            df[col] = s.astype(object).where(s.notna(), np.nan)
    return df


def feature_types(X):
    """Split feature names into numeric and categorical lists."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def missing_value_table(df):
    """Missing values per column after clean_categories()."""
    n = df.isna().sum()
    return pd.DataFrame({"Missing": n, "Percentage (%)": (n / len(df) * 100).round(2)})


# ------------------------------------------------------------------ preprocessing / models
def build_es_preprocessor(numeric, categorical):
    """Unfitted ColumnTransformer:
    numeric     -> median imputation -> standard scaling
    categorical -> most-frequent imputation -> one-hot encoding
                   (binary columns become a single 0/1 column)."""
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                                   sparse_output=False))]), categorical),
    ])


def clean_feature_names(names):
    """'cat__Polyuria_Yes' -> 'Polyuria = Yes', 'num__Age' -> 'Age'."""
    out = []
    for n in names:
        prefix, _, rest = n.partition("__")
        if prefix == "cat":
            base, _, level = rest.rpartition("_")
            rest = f"{base} = {level}"
        out.append(rest)
    return out


def get_es_classifiers(class_weight=None, scale_pos_weight=1.0, random_state=RANDOM_STATE):
    """The same five model families as for Dataset 1, with the same settings.

    class_weight="balanced" (and the matching scale_pos_weight for XGBoost)
    re-weights the classes; the notebook decides whether to use it with
    cross-validation on the training data."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=class_weight,
                                                  random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                                class_weight=class_weight, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight=class_weight, random_state=random_state,
                                                n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", scale_pos_weight=scale_pos_weight,
            random_state=random_state, n_jobs=-1)
    else:
        print("XGBoost is not installed - skipping it (pip install xgboost).")
    return models


def get_es_models(numeric, categorical, class_weight=None, scale_pos_weight=1.0,
                  random_state=RANDOM_STATE):
    """Model name -> unfitted Pipeline(preprocess -> classifier)."""
    classifiers = get_es_classifiers(class_weight, scale_pos_weight, random_state)
    return {name: Pipeline([("preprocess", build_es_preprocessor(numeric, categorical)),
                            ("model", clf)])
            for name, clf in classifiers.items()}
