"""Approach 1: extract color/texture features and train an SVM."""
from pathlib import Path

import numpy as np
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import features_from_path


def extract_matrix(image_paths):
    feature_rows = []
    for path in image_paths:
        feature_rows.append(features_from_path(path))
    return np.array(feature_rows, dtype=np.float32)


def fit(train, val, out_dir):
    # val is provided for consistency with CNN training, but the SVM
    # does not use validation labels when fitting its parameters.
    x_train = extract_matrix(train["path"])
    y_train = train["label"].to_numpy()

    svm = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=3.0, gamma="scale",
                    class_weight="balanced", probability=True, random_state=42))
    ])
    svm.fit(x_train, y_train)

    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    dump(svm, folder / "classical_svm.joblib")
    return svm


def predict(model, dataframe):
    features = extract_matrix(dataframe["path"])
    scores = model.predict_proba(features)
    labels = list(model.classes_)
    predictions = [labels[index] for index in scores.argmax(axis=1)]
    return predictions, scores


def load_model(path):
    # Never load a .joblib file from an unknown source.
    return load(path)
