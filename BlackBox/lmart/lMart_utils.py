import sys
from typing import Tuple

from lightgbm import LGBMRanker
from numpy import asarray, ndarray
from pandas import read_csv, DataFrame
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split, ParameterGrid
from tqdm import tqdm


class LMARTGridsearch:

    def __init__(self, path_dataset: str,
                 random_state: int = None,
                 split_size: Tuple[float, float] = (0.33, 0.33),
                 nDCG_at: int = 15):

        scores = read_csv(path_dataset)

        # Holdout splitting
        train, test = train_test_split(scores, test_size=split_size[0], random_state=random_state)
        train, valid = train_test_split(train, test_size=split_size[1], random_state=random_state)

        # sorting after the splitting
        self.train = train.sort_values(["qId", "kId"])
        self.valid = valid.sort_values(["qId", "kId"])
        self.test = test.sort_values(["qId", "kId"])

        # Preparing the datasets
        self.qIds_train = self.train.groupby("qId")["qId"].count().to_numpy()
        self.X_train, self.y_train = self.train.iloc[:, 2:13], self.train[["qId", "kId", "labels"]]

        self.qIds_val = self.valid.groupby("qId")["qId"].count().to_numpy()
        self.X_valid, self.y_valid = self.valid.iloc[:, 2:13], self.valid[["qId", "kId", "labels"]]

        self.qIds_test = self.test.groupby("qId")["qId"].count().to_numpy()
        self.X_test, self.y_test = self.test.iloc[:, 2:13], self.test[["qId", "kId", "labels"]]

        self.default_par = dict(
            objective="lambdarank",
            class_weight="balanced",
            metric="ndcg",
            importance_type="gain",
            force_row_wise=True,
            n_jobs=-1,
            verbose=-1
        )

        def log_output(eval_result):
            pass

        self.ranker_par = dict(
            X=self.X_train,
            y=self.y_train["labels"],
            group=self.qIds_train,
            eval_set=[(self.X_valid, self.y_valid["labels"])],
            eval_group=[self.qIds_val],
            eval_at=nDCG_at,
            callbacks=[log_output],

        )
        self.nDCG_at = nDCG_at
        return

    def eval_model(self, model: LGBMRanker, df: DataFrame = None, qIds: ndarray = None) -> float:

        df = self.valid if df is None else df
        n_qIds = len(self.qIds_val) if qIds is None else len(qIds)

        avg_nDCG = 0
        for _, v in df.groupby("qId"):
            tr, y = v.iloc[:, 2:13], asarray([v["labels"].to_numpy()])
            y_pred = asarray([model.predict(tr)])
            avg_nDCG += ndcg_score(y, y_pred, k=self.nDCG_at)
        return avg_nDCG / n_qIds

    def grid_search(self, hyperparameters: dict = None) -> Tuple:

        best_model_ = (None, None, -sys.maxsize)

        progress_bar = tqdm(ParameterGrid(hyperparameters))
        for conf in progress_bar:

            model = LGBMRanker(**self.default_par, **conf)
            model.fit(**self.ranker_par)

            avg_nDCG = self.eval_model(model)
            if avg_nDCG > best_model_[2]:
                best_model_ = (model, conf, avg_nDCG)

            progress_bar.set_postfix(nDCG=best_model_[2])
        return best_model_
