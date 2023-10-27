import sys

import numpy as np
from numpy.core.records import ndarray
from sklearn.metrics import ndcg_score
from numpy import asarray
from tqdm import tqdm
from sklearn.model_selection import train_test_split, ParameterGrid
from pandas import read_csv, DataFrame
from typing import Tuple

from Utils.Utils import GridSearch


class EBMGridSearch(GridSearch):
    def __init__(self, path_dataset: str,
                 task: str = "Regressor",
                 random_state: int = None,
                 split_size: Tuple[float, float] = (0.33, 0.33),
                 nDCG_at: int = 15):

        scores = read_csv(path_dataset)

        # Holdout splitting
        train, self.test = train_test_split(scores, test_size=split_size[0], random_state=random_state)
        self.train, self.valid = train_test_split(train, test_size=split_size[1], random_state=random_state)

        target = ["w_score"] if task == "Regressor" else ["labels"]
        self.X_train, self.y_train = self.train.iloc[:, 2:13], self.train[target]
        self.X_valid, self.y_valid = self.valid.iloc[:, 2:13], self.valid[target]
        self.X_test, self.y_test = self.test.iloc[:, 2:13], self.test[target]

        self.features_name = list(scores.iloc[:, 2:13].columns)
        self.default_par = dict(
            feature_names=self.features_name,
            n_jobs=-1,
            objective="rmse" if task == "Regressor" else "log_loss",
            exclude=[],
            feature_types=None,
            max_bins=256,
            max_interaction_bins=32,
            interactions=0,
            validation_size=0.15,
            outer_bags=8,
            inner_bags=0,
            greediness=0.0,
            smoothing_rounds=0,
            max_rounds=5000,
            early_stopping_rounds=50,
            early_stopping_tolerance=0.0001)

        self.nDCG_at = nDCG_at
        return

    def eval_model(self, model, df: DataFrame = None,
                   nDCG_at: list = None) -> dict:

        df = self.valid if df is None else df
        nDCG_at = [self.nDCG_at] if nDCG_at is None else nDCG_at
        avg_nDCG = np.zeros((len(nDCG_at)))

        n_groups = 0

        for _, v in df.groupby("qId"):
            v = v.sort_values("labels", ascending=False)

            features, target = v.iloc[:, 2:13].values, asarray([v["labels"].to_numpy()])
            y_pred = asarray([model.predict(features)])
            # Perform the nDCG for a specific job-offer and then sum it into cumulative nDCG
            for i, nDCG in enumerate(nDCG_at):
                avg_nDCG[i] += ndcg_score(target, y_pred, k=nDCG)
            n_groups += 1

        # dived by the number of jobs-offer to obtain the average.
        avg_nDCG /= n_groups
        results = {"nDCG@" + str(nDCG): round(avg_nDCG[i], 4) for i, nDCG in enumerate(nDCG_at)}
        return results

    def grid_search(self, EBMModel, hyperparameters: dict = None):

        best_model_: Tuple = (None, None, -sys.maxsize)

        progress_bar = tqdm(ParameterGrid(hyperparameters))
        for conf in progress_bar:

            model = EBMModel(**self.default_par, **conf)
            model.fit(self.X_train, self.y_train)
            avg_nDCG = self.eval_model(model)["nDCG@" + str(self.nDCG_at)]

            if avg_nDCG > best_model_[2]:
                best_model_ = (model, conf, avg_nDCG)
            progress_bar.set_postfix(nDCG_15_at=best_model_[2])
        return best_model_

    @staticmethod
    def pairwise_function(cuts: ndarray, contribution: ndarray, value: float) -> float:

        if value < cuts[0]:
            return contribution[0]
        for i in range(len(cuts)-1):
            if cuts[i] <= value <= cuts[i+1]:
                return contribution[i+1]
        if value > cuts[-1]:
            return contribution[-1]

    def explanation(self, model, index_feature: int, eps: float = 0.01) -> Tuple[ndarray, ndarray]:

        min_, max_ = model.feature_bounds_[index_feature]
        cuts = model.bins_[index_feature][0]
        contrib = model.term_scores_[index_feature][1:-1]

        x = np.arange(min_, max_, eps)
        y = [self.pairwise_function(cuts, contrib, v_) for v_ in x]
        return x, y
