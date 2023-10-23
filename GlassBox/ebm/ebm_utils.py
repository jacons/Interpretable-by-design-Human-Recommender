import sys
from typing import Tuple

from interpret.glassbox import ExplainableBoostingClassifier as EBClassifier
from numpy import asarray
from pandas import read_csv, DataFrame
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split, ParameterGrid
from tqdm import tqdm


class EBMGridSearch:
    def __init__(self, path_dataset: str,
                 random_state: int = None,
                 split_size: Tuple[float, float] = (0.33, 0.33),
                 nDCG_at: int = 15):

        scores = read_csv(path_dataset)

        # Holdout splitting
        train, self.test = train_test_split(scores, test_size=split_size[0], random_state=random_state)
        self.train, self.valid = train_test_split(train, test_size=split_size[1], random_state=random_state)

        self.X_train, self.y_train = self.train.iloc[:, 2:13], self.train[["labels"]]
        self.X_valid, self.y_valid = self.valid.iloc[:, 2:13], self.valid[["labels"]]
        self.X_test, self.y_test = self.test.iloc[:, 2:13], self.test[["labels"]]

        self.default_par = dict(
            feature_names=list(scores.iloc[:, 2:13].columns),
            n_jobs=-1,
            objective='log_loss',
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

    def eval_model(self, model: EBClassifier, df: DataFrame = None) -> float:

        df = self.valid if df is None else df

        avg_nDCG, n_groups = 0, 0

        for _, v in df.groupby("qId"):
            tr, y = v.iloc[:, 2:13].values, asarray([v["labels"].to_numpy()])
            y_pred = asarray([model.predict(tr)])
            avg_nDCG += ndcg_score(y, y_pred, k=self.nDCG_at)
            n_groups += 1

        return avg_nDCG / n_groups

    def grid_search(self, hyperparameters: dict = None):

        best_model_ = (None, None, -sys.maxsize)

        progress_bar = tqdm(ParameterGrid(hyperparameters))
        for conf in progress_bar:
            model = EBClassifier(**self.default_par, **conf)
            model.fit(self.X_train, self.y_train)

            avg_nDCG = self.eval_model(model)

            if avg_nDCG > best_model_[2]:
                best_model_ = (model, conf, avg_nDCG)
            progress_bar.set_postfix(nDCG=best_model_[2])
        return best_model_
