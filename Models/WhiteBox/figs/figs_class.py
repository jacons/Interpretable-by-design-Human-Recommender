import sys
from typing import Tuple

import numpy as np
from imodels import FIGSRegressor
from numpy import asarray
from pandas import DataFrame, read_json
from sklearn.metrics import ndcg_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from Models.grid_search_utils import GridSearch


class FIGS_class(GridSearch):

    def __init__(self, name: str, path: str = None, nDCG_at: int = 15):

        self.train = read_json(f"{path}{name}_dataset_tr.json")
        self.valid = read_json(f"{path}{name}_dataset_vl.json")
        self.test = read_json(f"{path}{name}_dataset_ts.json")

        self.train["relevance"] = self.train["relevance"].apply(lambda x: max(0, x))
        self.valid["relevance"] = self.valid["relevance"].apply(lambda x: max(0, x))
        self.test["relevance"] = self.test["relevance"].apply(lambda x: max(0, x))

        self.X_train, self.y_train = self.train.iloc[:, 5:].to_numpy(), self.train["relevance"].to_numpy()
        self.X_valid, self.y_valid = self.valid.iloc[:, 5:].to_numpy(), self.valid["relevance"].to_numpy()
        self.X_test, self.y_test = self.test.iloc[:, 5:].to_numpy(), self.test["relevance"].to_numpy()

        # features for the decision trees
        self.feature_name = list(self.train.iloc[:, 5:].columns)
        self.nDCG_at = nDCG_at
        return

    def eval_model(self, model, df: DataFrame = None,
                   nDCG_at: list = None) -> dict:
        """
        Custom evaluation function: the function groups by the "job-offers" and foreach set, it predicts
        the "regression score" that it uses to sort (by binned_relevance).
        After obtained nDCGs apply the average.
        """
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

    @staticmethod
    def split_list(all_configs, n):
        sublist_length = len(all_configs) // n
        result = [all_configs[i:i + sublist_length] for i in range(0, len(all_configs), sublist_length)]
        return result

    def grid_search(self, hyperparameters: dict = None, ):

        # keep the current: (best_model, best_params, best nDCG)
        best_model_: Tuple = (None, None, -sys.maxsize)

        # explore all possible combinations of hyperparameters
        progress_bar = tqdm(ParameterGrid(hyperparameters), desc="Finding the best model")
        for conf in progress_bar:

            model = FIGSRegressor(**conf)
            model.fit(self.X_train, self.y_train, self.feature_name)
            avg_nDCG = self.eval_model(model)["nDCG@" + str(self.nDCG_at)]

            # if the model is better respect to the previous one, it updates the tuple
            if avg_nDCG > best_model_[2]:
                best_model_ = (model, conf, avg_nDCG)
            progress_bar.set_postfix(nDCG=best_model_[2])

        return best_model_
