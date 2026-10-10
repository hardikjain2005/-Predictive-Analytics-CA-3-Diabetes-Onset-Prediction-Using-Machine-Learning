"""Model definitions and training helpers.

Every model is wrapped in an imbalanced-learn Pipeline:

    median imputation -> standard scaling -> SMOTE -> classifier

Because SMOTE sits *inside* the pipeline, it is only applied when the
pipeline is fitted, i.e. to the training data (or to the training folds
during cross-validation). It is never applied when predicting, so the test
set keeps its real class distribution.
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def get_classifiers(random_state=RANDOM_STATE):
    """The five classifiers compared in the project, with simple,
    reasonable settings (no exhaustive hyperparameter search)."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def build_pipeline(classifier, use_smote=True, random_state=RANDOM_STATE):
    """Wrap a classifier with imputation, scaling and (optionally) SMOTE.

    Scaling is essential for Logistic Regression and KNN. Tree models do not
    need it, but it does not change their splits and it keeps SMOTE's
    nearest-neighbour search on comparable feature scales, so the same
    pipeline is used for every model.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", classifier))
    return ImbPipeline(steps)


def get_models(random_state=RANDOM_STATE):
    """Dictionary of model name -> unfitted SMOTE pipeline."""
    return {
        name: build_pipeline(clf, random_state=random_state)
        for name, clf in get_classifiers(random_state).items()
    }


def train_models(models, X_train, y_train):
    """Fit every pipeline on the training data and return them."""
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained: {name}")
    return models


def cross_validate_models(models, X_train, y_train, cv_splits=5,
                          random_state=RANDOM_STATE):
    """Stratified k-fold cross-validation on the TRAINING data only.

    Returns mean and standard deviation of each metric per model. This gives
    a more stable estimate than a single test split and never touches the
    test set.
    """
    import pandas as pd

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                         random_state=random_state)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv,
                                scoring=scoring)
        row = {"Model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"CV {metric} mean"] = values.mean()
            row[f"CV {metric} std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")
