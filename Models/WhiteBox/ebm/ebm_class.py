import sys
from typing import Tuple

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from numpy import asarray
from pandas import DataFrame, read_json
from sklearn.metrics import ndcg_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from Models.grid_search_utils import GridSearch


class EBM_class(GridSearch):
    def __init__(self, name: str, path: str = None, nDCG_at: int = 15):

        self.train = read_json(f"{path}{name}_dataset_tr.json")
        self.valid = read_json(f"{path}{name}_dataset_vl.json")
        self.test = read_json(f"{path}{name}_dataset_ts.json")

        self.train["relevance"] = self.train["relevance"].apply(lambda x: max(0, x))
        self.valid["relevance"] = self.valid["relevance"].apply(lambda x: max(0, x))
        self.test["relevance"] = self.test["relevance"].apply(lambda x: max(0, x))

        self.X_train, self.y_train = self.train.iloc[:, 5:], self.train["relevance"]
        self.X_valid, self.y_valid = self.valid.iloc[:, 5:], self.valid["relevance"]
        self.X_test, self.y_test = self.test.iloc[:, 5:], self.test["relevance"]

        self.features_name = list(self.train.iloc[:, 5:].columns)
        self.default_par = dict(
            feature_names=self.features_name,
            n_jobs=-1,
            objective="rmse",
            exclude=[],
            interactions=0,
            max_interaction_bins=32,
            validation_size=0.15,
            inner_bags=0,
            greediness=0.0,
            smoothing_rounds=0,
            max_rounds=8000,
            early_stopping_rounds=50,
            early_stopping_tolerance=0.0001)

        self.nDCG_at = nDCG_at

    def eval_model(self, model, df: DataFrame = None, nDCG_at: list = None) -> dict:

        df = self.valid if df is None else df
        nDCG_at = [self.nDCG_at] if nDCG_at is None else nDCG_at
        avg_nDCG = np.zeros((len(nDCG_at)))

        n_groups = 0

        for _, v in df.groupby("qId"):

            features, target = v.iloc[:, 5:].values, asarray([v["relevance"].to_numpy()])
            y_pred = asarray([model.predict(features)])
            # Perform the nDCG for a specific job-offer and then sum it into cumulative nDCG
            for i, nDCG in enumerate(nDCG_at):
                avg_nDCG[i] += ndcg_score(target, y_pred, k=nDCG)
            n_groups += 1

        # dived by the number of jobs-offer to obtain the average.
        avg_nDCG /= n_groups
        results = {"nDCG@" + str(nDCG): round(avg_nDCG[i], 4) for i, nDCG in enumerate(nDCG_at)}
        return results

    def grid_search(self, hyperparameters: dict = None):

        best_model_: Tuple = (None, None, -sys.maxsize)

        progress_bar = tqdm(ParameterGrid(hyperparameters), desc="Finding the best model")
        for conf in progress_bar:

            model = ExplainableBoostingRegressor(**self.default_par, **conf)
            model.fit(self.X_train, self.y_train)
            avg_nDCG = self.eval_model(model)["nDCG@" + str(self.nDCG_at)]

            if avg_nDCG > best_model_[2]:
                best_model_ = (model, conf, avg_nDCG)
            progress_bar.set_postfix(nDCG_15_at=best_model_[2])
        return best_model_
