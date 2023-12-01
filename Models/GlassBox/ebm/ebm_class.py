import sys

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

from numpy import ndarray
from sklearn.metrics import ndcg_score
from numpy import asarray
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from pandas import read_csv, DataFrame
from typing import Tuple

from Models.grid_search_utils import GridSearch


class EBM_class(GridSearch):
    def __init__(self, name: str, path: str = None, nDCG_at: int = 15):

        self.train = read_csv(f"{path}{name}_dataset_tr.csv")
        self.valid = read_csv(f"{path}{name}_dataset_vl.csv")
        self.test = read_csv(f"{path}{name}_dataset_ts.csv")

        self.train["w_score"] = self.train["w_score"].apply(lambda x: max(0, x))
        self.valid["w_score"] = self.valid["w_score"].apply(lambda x: max(0, x))
        self.test["w_score"] = self.test["w_score"].apply(lambda x: max(0, x))

        self.X_train, self.y_train = self.train.iloc[:, 5:], self.train["w_score"]
        self.X_valid, self.y_valid = self.valid.iloc[:, 5:], self.valid["w_score"]
        self.X_test, self.y_test = self.test.iloc[:, 5:], self.test["w_score"]

        self.features_name = list(self.train.iloc[:, 5:].columns)
        self.default_par = dict(
            feature_names=self.features_name,
            n_jobs=-1,
            objective="rmse",
            exclude=[],
            interactions=0,
            feature_types=None,
            max_interaction_bins=32,
            validation_size=0.15,
            inner_bags=0,
            greediness=0.0,
            smoothing_rounds=0,
            max_rounds=8000,
            early_stopping_rounds=50,
            early_stopping_tolerance=0.0001)

        self.nDCG_at = nDCG_at

        self.piecewise_functions = []

    def eval_model(self, model, df: DataFrame = None,
                   nDCG_at: list = None) -> dict:

        df = self.valid if df is None else df
        nDCG_at = [self.nDCG_at] if nDCG_at is None else nDCG_at
        avg_nDCG = np.zeros((len(nDCG_at)))

        n_groups = 0

        for _, v in df.groupby("qId"):

            features, target = v.iloc[:, 5:].values, asarray([v["w_score"].to_numpy()])
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

    @staticmethod
    def pairwise_function(cuts: ndarray, contribution: ndarray, value: float):

        if value < cuts[0]:
            return contribution[0]
        for i in range(len(cuts) - 1):
            if cuts[i] <= value <= cuts[i + 1]:
                return contribution[i + 1]
        if value > cuts[-1]:
            return contribution[-1]

    def build_piecewise_functions(self, model: ExplainableBoostingRegressor):

        for idx, feature in enumerate(model.feature_names):
            min_, max_ = model.feature_bounds_[idx]
            fun = PiecewiseFunction(feature,
                                    model.bins_[idx][0],
                                    model.term_scores_[idx][1:-1],
                                    model.standard_deviations_[idx][1:-1],
                                    min_, max_)
            self.piecewise_functions.append(fun)

    def show_piecewise_functions(self, model: ExplainableBoostingRegressor) -> list[DataFrame]:
        return [self.piecewise_functions[idx].show_function() for idx, _ in enumerate(model.feature_names)]


class PiecewiseFunction:
    def __init__(self, name: str, cuts: np.ndarray, contrib: np.ndarray, std_dev: np.ndarray,
                 min_: float, max_: float):
        self.name = name
        self.cuts = cuts.tolist()
        self.contrib = contrib.tolist()
        self.std_dev = std_dev.tolist()
        self.min_ = min_
        self.max_ = max_

    def get_result(self, x: float) -> Tuple[float, float]:
        output = None

        if x <= self.cuts[0]:
            output = self.contrib[0], self.std_dev[0]
        elif x >= self.cuts[-1]:
            output = self.contrib[-1], self.std_dev[-1]
        else:
            for i in range(len(self.cuts) - 1):
                if self.cuts[i] <= x <= self.cuts[i + 1]:
                    output = self.contrib[i + 1], self.std_dev[i + 1]
                    break

        return output

    def show_function(self) -> DataFrame:
        line_space = np.arange(self.min_, self.max_, 0.01)

        x, y_lowers, y, y_uppers = [], [], [], []

        for x_ in line_space:
            y_, y_std = self.get_result(x_)

            x.append(x_)
            y_lowers.append(y_ - y_std)
            y.append(y_)
            y_uppers.append(y_ + y_std)

        return DataFrame({"x": x, "lower": y_lowers, "y": y, "upper": y_uppers})
