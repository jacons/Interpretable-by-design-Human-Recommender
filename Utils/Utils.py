import json
import os
import pickle
from typing import Tuple, Any

import numpy as np
from lightgbm import LGBMRanker
from numpy import ndarray, asarray
from pandas import DataFrame
from sklearn.metrics import ndcg_score


class GridSearch:
    @staticmethod
    def save_model(model: LGBMRanker, name: str = "model"):
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        with open('saved_models/' + name + ".pkl", 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(name: str = "model") -> Tuple[Any, dict]:
        with open('saved_models/' + name + ".pkl", 'rb') as file:
            model = pickle.load(file)
        return model

    @staticmethod
    def other_eval(df: DataFrame = None, qIds: ndarray = None, nDCG_at: list = None) -> dict:

        n_qIds = len(qIds)
        avg_nDCG = np.zeros((3, len(nDCG_at)))

        for _, v in df.groupby("qId"):
            v = v.sort_values("labels", ascending=False)

            _, y = v.iloc[:, 2:13], asarray([v["labels"].to_numpy()])

            random_lambdas = asarray([v.sample(frac=1)["labels"].to_numpy()])
            perfect_lambdas = asarray([v.sort_values("labels", ascending=False)["labels"].to_numpy()])
            worst_lambdas = asarray([v.sort_values("labels")["labels"].to_numpy()])

            for i, nDCG in enumerate(nDCG_at):
                avg_nDCG[0, i] += ndcg_score(y, random_lambdas, k=nDCG)
                avg_nDCG[1, i] += ndcg_score(y, perfect_lambdas, k=nDCG)
                avg_nDCG[2, i] += ndcg_score(y, worst_lambdas, k=nDCG)

        # dived by the number of jobs-offer to obtain the average.
        avg_nDCG /= n_qIds

        results = dict(
            random_permutation={"nDCG@" + str(nDCG): round(avg_nDCG[0][i], 4) for i, nDCG in enumerate(nDCG_at)},
            perfect_nDCG={"nDCG@" + str(nDCG): round(avg_nDCG[1][i], 4) for i, nDCG in enumerate(nDCG_at)},
            worste_nDCG={"nDCG@" + str(nDCG): round(avg_nDCG[2][i], 4) for i, nDCG in enumerate(nDCG_at)},
        )
        return results
