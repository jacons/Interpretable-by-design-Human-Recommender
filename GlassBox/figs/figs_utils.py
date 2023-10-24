import sys
from multiprocessing import Process, Manager
from sklearn.metrics import ndcg_score
from numpy import asarray, empty
from imodels import FIGSRegressor
from tqdm import tqdm
from sklearn.model_selection import train_test_split, ParameterGrid
from pandas import read_csv, DataFrame
from typing import Tuple


class FIGSGridSearch:
    def __init__(self, path_dataset: str,
                 random_state: int = None,
                 split_size: Tuple[float, float] = (0.33, 0.33),
                 nDCG_at: int = 15):

        scores = read_csv(path_dataset)

        # features for the decision trees
        self.feature_name = list(scores.iloc[:, 2:13].columns)

        # Holdout splitting
        train, self.test = train_test_split(scores, test_size=split_size[0], random_state=random_state)
        self.train, self.valid = train_test_split(train, test_size=split_size[1], random_state=random_state)

        self.X_train, self.y_train = asarray(self.train.iloc[:, 2:13]), self.train[["w_score"]].values.ravel()
        self.X_valid, self.y_valid = asarray(self.valid.iloc[:, 2:13]), self.valid[["w_score"]].values.ravel()
        self.X_test, self.y_test = asarray(self.test.iloc[:, 2:13]), self.test[["w_score"]].values.ravel()

        self.nDCG_at = nDCG_at
        return

    def eval_model(self, model: FIGSRegressor, df: DataFrame = None, nDCG_at: int = -1) -> float:
        """
        Custom evaluation function: the function groups by the "job-offers" and foreach set, it predicts
        the "regression score" that it uses to sort (by relevance).
        After obtained nDCGs apply the average.
        """
        df = self.valid if df is None else df
        nDCG_at = self.nDCG_at if nDCG_at == -1 else nDCG_at

        avg_nDCG, n_groups = 0, 0

        for _, v in df.groupby("qId"):
            tr, y = v.iloc[:, 2:13].values, asarray([v["labels"].to_numpy()])
            y_pred = asarray([model.predict(tr)])
            avg_nDCG += ndcg_score(y, y_pred, k=nDCG_at)
            n_groups += 1

        return avg_nDCG / n_groups

    def fit(self, **param) -> FIGSRegressor:
        model = FIGSRegressor(**param)
        model.fit(self.X_train, self.y_train, self.feature_name)
        return model

    @staticmethod
    def split_list(all_configs, n):
        sublist_length = len(all_configs) // n
        result = [all_configs[i:i + sublist_length] for i in range(0, len(all_configs), sublist_length)]
        return result

    def parallel_gridsearch_routine(self, params: list, worker_id: int, sycDict: dict):

        print("Start worker ", worker_id)
        best_model_: Tuple = (None, None, -sys.maxsize)

        for conf in params:
            model = FIGSRegressor(**conf)
            model.fit(self.X_train, self.y_train, self.feature_name)

            avg_nDCG = self.eval_model(model)

            if avg_nDCG > best_model_[2]:
                best_model_ = (model, conf, avg_nDCG)

        sycDict[worker_id] = best_model_
        print("End worker ", worker_id)
        return

    def parallel_gridsearch(self, hyperparameters: dict = None, jobs: int = 2) -> Tuple:

        best_models = Manager().dict()  # define a synchronized and shared variable.
        grid_list = self.split_list(list(ParameterGrid(hyperparameters)), jobs)

        workers = empty(jobs, dtype=Process)
        for idx, params in enumerate(grid_list):
            workers[idx] = Process(target=self.parallel_gridsearch_routine, args=(params, idx, best_models))
            workers[idx].start()

        for idx, _ in enumerate(grid_list):
            workers[idx].join()

        print(best_models)

        best_model = None
        for v in best_models.values():
            if best_model is None:
                best_model = v
            elif v[2] > best_model[2]:
                best_model = v

        print(best_model)
        return best_model  # best_model_

    def grid_search(self, hyperparameters: dict = None):

        # keep the current: (best_model, best_params, best nDCG)
        best_model_: Tuple = (None, None, -sys.maxsize)

        # explore all possible combinations of hyperparameters
        progress_bar = tqdm(ParameterGrid(hyperparameters))
        for conf in progress_bar:

            model = self.fit(**conf)
            avg_nDCG = self.eval_model(model)

            # if the model is better respect to the previous one, it updates the tuple
            if avg_nDCG > best_model_[2]:
                best_model_ = (model, conf, avg_nDCG)
            progress_bar.set_postfix(nDCG=best_model_[2])

        return best_model_
