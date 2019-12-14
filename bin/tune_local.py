#!/usr/bin/env python
# %%
from ray import tune
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


# %%
def train_rf(config):
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    for i in range(6):
        clf = RandomForestClassifier(max_depth=config["max_depth"], random_state=0)
        y_pred = cross_val_predict(clf, X, y, cv=5)
        auc = roc_auc_score(y_true=y, y_score=y_pred)
        tune.track.log(auc=auc)


# %%
analysis = tune.run(train_rf, config={"max_depth": tune.grid_search([1, 4, 20])})

print("Best config: ", analysis.get_best_config(metric="auc"))

analysis.fetch_trial_dataframes()

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
print(df)

# %%
