#!/usr/bin/env python
# Example for ASHA scheduler Early, random search with stopping less promising runs
# See: https://ray.readthedocs.io/en/latest/tune-tutorial.html#early-stopping-with-asha

# %%
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from tune_demo.train import get_iris_data, rf_cv


# %%
def eval_model(config):
    X, y = get_iris_data()
    for i in range(5):
        acc = rf_cv(config, X, y)
        tune.track.log(acc=acc)


# %%
config = {
    "max_depth": tune.sample_from(lambda spec: np.random.randint(1, 21)),
    "n_estimators": tune.sample_from(lambda spec: np.random.randint(10, 1001)),
}

analysis = tune.run(
    eval_model,
    num_samples=12,
    scheduler=ASHAScheduler(metric="acc", mode="max"),
    config=config
)

print("Best config: ", analysis.get_best_config(metric="acc"))
# config = {'max_depth': 4, 'n_estimators': 1000}

# Get a dataframe for analyzing trial results.
print(analysis.dataframe())

# %%
