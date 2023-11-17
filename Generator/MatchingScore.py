from typing import Tuple

import numpy as np
from numpy.random import normal
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Class_utils import JobGraph, FitnessFunctions


class MatchingScore:
    def __init__(self,
                 job_graph: JobGraph, fitness_functions: FitnessFunctions, bins: int, weight: np.ndarray,
                 noise: Tuple[float], split_size: Tuple[float], split_seed: int):

        self.job_graph = job_graph
        self.fitness = fitness_functions
        self.weights = self._normalize_weights(weight)
        self.noise = noise  # mean and stddev
        self.bins = bins  # Number of relevance's levels
        self.split_size = split_size
        self.split_seed = split_seed

    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        return weights / weights.sum()

    def score_function(self, offers: DataFrame, curricula: DataFrame, name: str = None) -> DataFrame:
        dataset = self._generate_fitness_score(offers, curricula)
        dataset = self._compute_score(dataset)
        dataset = self._create_relevance_labels(dataset)
        _ = self._split_and_save_datasets(dataset, name)
        self._save_output(dataset, name)
        return dataset.set_index(keys=["qId", "kId"])

    def _generate_fitness_score(self, offers: DataFrame, curricula: DataFrame) -> DataFrame:
        dataset = []
        bar = tqdm(offers.itertuples(), total=len(offers), desc="Generating the fitness scores")
        for offer in bar:
            curricula_ = curricula[curricula.index.get_level_values(0) == offer[0]]
            for cv in curricula_.itertuples():
                dataset.append(self.fitness.fitness(offer, cv))
            bar.set_postfix(qId=offer[0])

        dataset = DataFrame(data=dataset, dtype=np.float32).astype({"qId": "int", "kId": "int"})
        return dataset

    def _compute_score(self, dataset: DataFrame) -> DataFrame:
        features = dataset.iloc[:, 2:]

        # Simple sum
        dataset["score"] = features.sum(axis=1)

        def target_fun(fitness_vector):
            result = np.dot(fitness_vector, self.weights) + normal(self.noise[0], self.noise[1])
            return max(0, result)

        # Weighted sum
        dataset['w_score'] = features.apply(target_fun, axis=1)

        return dataset

    def _create_relevance_labels(self, dataset: DataFrame) -> DataFrame:
        intervals, edges = np.histogram(dataset.sort_values("w_score", ascending=False)["w_score"].to_numpy(),
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

        dataset["relevance"] = dataset['w_score'].apply(score2label)

        rest_columns = [c for c in dataset.columns if c not in ["qId", "kId", "score", "w_score", "relevance"]]
        dataset = dataset.loc[:, ["qId", "kId", "score", "w_score", "relevance"] + rest_columns]

        return dataset

    @staticmethod
    def _save_output(dataset: DataFrame, name: str) -> None:
        if name is not None:
            dataset.to_csv(f"../outputs/scores/{name}_dataset.csv", index=False)

    def _split_and_save_datasets(self, dataset: DataFrame, name: str) -> Tuple[DataFrame, DataFrame, DataFrame]:
        train, test = train_test_split(dataset, test_size=self.split_size[0], random_state=self.split_seed)
        train, valid = train_test_split(train, test_size=self.split_size[1], random_state=self.split_seed)

        if name is not None:
            train.to_csv(f"../outputs/scores/{name}_dataset_tr.csv", index=False)
            valid.to_csv(f"../outputs/scores/{name}_dataset_vl.csv", index=False)
            test.to_csv(f"../outputs/scores/{name}_dataset_ts.csv", index=False)

        return train, valid, test
