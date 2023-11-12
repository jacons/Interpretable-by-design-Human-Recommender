import sys
from typing import Tuple

import numpy as np
from lightgbm import LGBMRanker
from numpy import asarray, ndarray
from pandas import read_csv, DataFrame
from sklearn.metrics import ndcg_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from Models.grid_search_utils import GridSearch


class LGBMRanker_class(GridSearch):

    def __init__(self, name: str, path: str = None, nDCG_at: int = 15):
        self.train = read_csv(f"{path}{name}_dataset_tr.csv")
        self.valid = read_csv(f"{path}{name}_dataset_vl.csv")
        self.test = read_csv(f"{path}{name}_dataset_ts.csv")

        # sorting after the splitting
        self.train.sort_values(["qId", "kId"], inplace=True)
        self.valid.sort_values(["qId", "kId"], inplace=True)
        self.test.sort_values(["qId", "kId"], inplace=True)

        # Preparing the datasets
        self.qIds_train = self.train.groupby("qId")["qId"].count().to_numpy()
        self.X_train, self.y_train = self.train.iloc[:, 5:], self.train[["qId", "kId", "relevance"]]

        self.qIds_valid = self.valid.groupby("qId")["qId"].count().to_numpy()
        self.X_valid, self.y_valid = self.valid.iloc[:, 5:], self.valid[["qId", "kId", "relevance"]]

        self.qIds_test = self.test.groupby("qId")["qId"].count().to_numpy()
        self.X_test, self.y_test = self.test.iloc[:, 5:], self.test[["qId", "kId", "relevance"]]

        def log_output(r):
            # callback function (used to avoid logging during the grid-search)
            pass

        self.default_par = dict(  # default parameters pt.1
            objective="lambdarank",
            class_weight="balanced",
            metric="ndcg",
            importance_type="gain",
            force_row_wise=True,
            n_jobs=-1,
            verbose=-1
        )
        self.ranker_par = dict(  # default ranker parameters (used in fitting) pt.2
            X=self.X_train,
            y=self.y_train["relevance"],
            group=self.qIds_train,
            eval_set=[(self.X_valid, self.y_valid["relevance"])],
            eval_group=[self.qIds_valid],
            eval_at=nDCG_at,
            callbacks=[log_output]
        )
        self.nDCG_at = nDCG_at
        return

    def grid_search(self, hyperparameters: dict = None) -> Tuple:

        # keep the current: (best_model, best_params, best nDCG)
        best_model_: Tuple = (None, None, -sys.maxsize)

        # explore all possible combinations of hyperparameters
        progress_bar = tqdm(ParameterGrid(hyperparameters))
        for conf in progress_bar:

            model = self.fit(**conf)
            avg_nDCG = self.eval_model(model)["nDCG@" + str(self.nDCG_at)]

            # if the model is better respect to the previous one, it updates the tuple
            if avg_nDCG > best_model_[2]:
                best_model_ = (model, conf, avg_nDCG)

            progress_bar.set_postfix(nDCG_15=best_model_[2])
        return best_model_

    def fit(self, **conf) -> LGBMRanker:
        model = LGBMRanker(**self.default_par, **conf)
        model.fit(**self.ranker_par)
        return model

    def eval_model(self, model: LGBMRanker, dt: DataFrame = None,
                   qIds: ndarray = None, nDCG_at: list = None) -> dict:
        """
        Custom evaluation function: the function groups by the "job-offers" and foreach set, it predicts
        the "lambdas" that it uses to sort (by relevance).
        After obtained nDCGs apply the average.
        """
        dt = self.valid if dt is None else dt
        n_qIds = len(self.qIds_valid) if qIds is None else len(qIds)
        nDCG_at = [self.nDCG_at] if nDCG_at is None else nDCG_at
        avg_nDCG = np.zeros((len(nDCG_at)))

        for _, v in dt.groupby("qId"):

            features, target = v.iloc[:, 5:], asarray([v["w_score"].to_numpy()])
            lambdas = asarray([model.predict(features)])  # predict lambdas

            # Perform the nDCG for a specific job-offer and then sum it into cumulative nDCG
            for i, nDCG in enumerate(nDCG_at):
                avg_nDCG[i] += ndcg_score(target, lambdas, k=nDCG)

        # dived by the number of jobs-offer to obtain the average.
        avg_nDCG /= n_qIds
        results = {"nDCG@" + str(nDCG): round(avg_nDCG[i], 4) for i, nDCG in enumerate(nDCG_at)}
        return results
