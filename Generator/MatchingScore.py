import os
from typing import Tuple

import numpy as np
from numpy.random import normal
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from Class_utils import FitnessFunctions


class MatchingScore:
    def __init__(self, fitness_functions: FitnessFunctions, bins: int, weight: dict,
                 noise: Tuple[float], split_size: Tuple[float], split_seed: int):

        self.fitness = fitness_functions
        self.weights = self.normalize_weights(weight)
        self.noise = noise  # mean and stddev
        self.bins = bins  # Number of binned_relevance's levels
        self.split_size = split_size
        self.split_seed = split_seed

    @staticmethod
    def normalize_weights(weights: dict) -> np.ndarray:
        ground_truth = np.asarray(list(weights.values()))
        return ground_truth

    def score_function(self, offers: DataFrame, curricula: DataFrame, path: str = None, name: str = None) -> DataFrame:
        dataset = self.fitness.generate_fitness_score(offers, curricula)
        dataset = self._compute_score(dataset)
        dataset = self._create_binned_score_labels(dataset)
        _ = self._split_and_save_datasets(dataset, path, name)
        self._save_output(dataset, path, name)
        return dataset.set_index(keys=["qId", "kId"])

    def _compute_score(self, dataset: DataFrame) -> DataFrame:
        features = dataset.iloc[:, 3:]

        def target_fun(fitness_vector):
            result = np.dot(fitness_vector, self.weights) + normal(self.noise[0], self.noise[1])
            return result

        # Weighted sum
        dataset['relevance'] = features.apply(target_fun, axis=1)
        dataset['relevance'] += abs(dataset['relevance'].min())

        return dataset

    def _create_binned_score_labels(self, dataset: DataFrame) -> DataFrame:
        intervals, edges = np.histogram(dataset.sort_values("relevance", ascending=False)["relevance"].to_numpy(),
                                        bins=self.bins)
        score2inter = {i: (edges[i], edges[i + 1]) for i in range(len(intervals))}

        def score2label(score_value: float) -> int:
            if score_value <= score2inter[0][0]:
                return 0
            for i, (v_min, v_max) in score2inter.items():
                if v_min <= score_value < v_max:
                    return i
            if score_value >= score2inter[self.bins - 1][1]:
                return self.bins - 1

        dataset["binned_relevance"] = dataset['relevance'].apply(score2label)

        rest_columns = [c for c in dataset.columns if c not in ["qId", "kId", "relevance", "binned_relevance"]]
        dataset = dataset.loc[:, ["qId", "kId", "relevance", "binned_relevance"] + rest_columns]

        return dataset

    @staticmethod
    def _save_output(dataset: DataFrame, path: str = None, name: str = None) -> None:
        if name is not None and path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            dataset.to_json(f"{path}/{name}_dataset.json", indent=2, orient="records")

    def _split_and_save_datasets(self, dataset: DataFrame, path: str = None,
                                 name: str = None) -> Tuple[DataFrame, DataFrame, DataFrame]:

        train, test = train_test_split(dataset, test_size=self.split_size[0], random_state=self.split_seed)
        train, valid = train_test_split(train, test_size=self.split_size[1], random_state=self.split_seed)

        if name is not None and path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            train.to_json(f"{path}/{name}_dataset_tr.json", indent=2, orient="records")
            valid.to_json(f"{path}/{name}_dataset_vl.json", indent=2, orient="records")
            test.to_json(f"{path}/{name}_dataset_ts.json", indent=2, orient="records")

        return train, valid, test
