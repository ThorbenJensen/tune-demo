#!/usr/bin/env python
# %%
from hyperopt import hp
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from tune_demo.train import get_iris_data, rf_cv


# %%
def eval_model(config):
    X, y = get_iris_data()
    for i in range(3):
        acc = rf_cv(config, X, y)
        tune.track.log(acc=acc)


# %%
space = {
    "max_depth": hp.uniformint("max_depth", 1, 20),
    "n_estimators": hp.uniformint("n_estimators", 10, 1000),
}

hyperopt_search = HyperOptSearch(space=space, max_concurrent=3, metric="acc", gamma=0.2)

analysis = tune.run(eval_model, num_samples=30, search_alg=hyperopt_search)

print("Best config: ", analysis.get_best_config(metric="acc"))
# config = {'max_depth': 4, 'n_estimators': 1000}

# Get a dataframe for analyzing trial results.
print(analysis.dataframe())

# %%
