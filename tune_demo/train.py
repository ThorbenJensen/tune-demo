"""Training loops"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict


def get_iris_data():
    iris = load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    return X, y


def rf_cv(config, X, y):
    clf = RandomForestClassifier(
        max_depth=config["max_depth"], n_estimators=config["n_estimators"],
    )
    y_pred = cross_val_predict(clf, X, y, cv=5)
    acc = accuracy_score(y_true=y, y_pred=y_pred)
    return acc
