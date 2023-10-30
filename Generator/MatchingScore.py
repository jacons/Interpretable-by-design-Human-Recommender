from itertools import product
from typing import Tuple

import numpy as np
from numpy.random import normal
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Generator.JobGenerator import JobGenerator
from Generator.SkillGraph import SkillGraph


class MatchingScore:
    def __init__(self,
                 jobGenerator: JobGenerator,
                 cities_dist: str,
                 bins: int,
                 weight: np.ndarray,
                 noise: Tuple[float],
                 split_size: Tuple[float],
                 split_seed: int):

        self.bins = bins  # Number of relevances
        self.noise = noise  # mean and stddev
        self.split_size = split_size
        self.split_seed = split_seed

        self.skillGraph = SkillGraph(jobGenerator.jobs)

        self.education_ranks = dict(zip(jobGenerator.education["Education"], jobGenerator.education["Rank"]))
        self.len_ed_rank = len(self.education_ranks)

        self.distance = read_csv(cities_dist, index_col=[0, 1], skipinitialspace=True)
        self.max_distance = self.distance["D"].max()

        self.lvl2value = {t[1]: t[0] for t in jobGenerator.languages_level.itertuples()}

        self.weights = self.normalize_weights(weight)

    @staticmethod
    def normalize_weights(weights: np.ndarray):
        return weights / weights.sum()

    def educationScore(self, offer: str, cv: str) -> int:
        edu_cv = self.education_ranks.get(cv)
        edu_offer = self.education_ranks.get(offer)

        # Education max 1, min 0
        return 1 if edu_cv >= edu_offer else 1 - (edu_offer - edu_cv) / self.len_ed_rank

    def cityScore(self, cityA: str, cityB: str, range_: int) -> float:

        if cityA == cityB:
            return 1

        s_cities = sorted([cityA, cityB])
        dist = self.distance.loc[(s_cities[0], s_cities[1])].values[0]

        return 1 if dist < range_ else 1 - (dist - range_) / self.max_distance

    def skillScore(self, offer: list, cv: list) -> float:

        offer = set([x for x in offer if x != "-"])
        cv = set([x for x in cv if x != "-"])

        n_offer_skil = len(offer)

        intersect = list(offer & cv)
        score = len(intersect) / len(offer)  # "1" perfect, "0" bad

        for i_ in intersect:
            offer.remove(i_)
            cv.remove(i_)

        plus = 0
        if len(offer) > 0 & len(cv) > 0:
            for job_sk_ in offer:
                paths = []
                for cv_sk_ in cv:
                    n_hops = self.skillGraph.shortest_path(cv_sk_, job_sk_) - 1
                    paths.append(n_hops)
                plus += 1 / min(paths)
            plus = plus / n_offer_skil

        return score + plus

    @staticmethod
    def softSkillScore(offer: list, cv: list) -> float:

        offer = set([x for x in offer if x != "-"])
        cv = set([x for x in cv if x != "-"])

        if len(offer) == 0:
            return 0
        return len(offer & cv) / len(offer)

    @staticmethod
    def certificateScore(offer: list, cv: list) -> float:

        offer = set([x for x in offer if x != "-"])
        cv = set([x for x in cv if x != "-"])

        if len(offer) == 0:
            return 0
        return len(offer & cv) / len(offer)

    @staticmethod
    def ageScore(cv: int, v_min: int, v_max: int, slope: int = 7) -> float:
        # Piecewise function to score. Max 1 Min 0

        if int(v_min <= cv <= v_max):
            return 1
        if cv > v_max:
            return max((-cv + v_max) / slope + 1, 0)
        return max((cv - v_min) / slope + 1, 0)

    def languageScore(self, offer: list, cv: list) -> float:

        offer_, cv_ = [], []
        for i_ in offer:
            if i_ != "-":
                offer_.append(i_.split(" - "))

        for i_ in cv:
            if i_ != "-":
                cv_.append(i_.split(" - "))

        score = 0
        for a, b in product(offer_, cv_):
            if a[0] == b[0]:
                lvl_of = self.lvl2value[a[1]]
                lvl_cv = self.lvl2value[b[1]]
                score += 1 if lvl_of <= lvl_cv else 1 / (lvl_of - lvl_cv)

        return score / len(offer_)

    @staticmethod
    def experienceScore(offer: int, cv: int) -> float:
        # Salary max 1, min 0
        return 1 if cv >= offer else 0 if cv <= 0 else max((cv - offer) / 8 + 1, 0)

    @staticmethod
    def salaryScore(offer: int, cv: int) -> float:
        # Salary max 1, min 0
        return 1 if cv <= offer else max((offer - cv) / 500 + 1, 0)

    def scoreFunction(self, offers: DataFrame, curricula: DataFrame, output_file: str = None):
        combinations = list(product(offers.itertuples(), curricula.itertuples()))
        score = np.zeros((len(combinations), 13), dtype=np.float32)

        for idx, (offer, cv) in enumerate(tqdm(combinations)):
            offer_skills = [offer[4], offer[5], offer[6], offer[7], offer[8]]
            cv_skills = [cv[5], cv[6], cv[7], cv[8], cv[9]]
            offer_s_skills = [offer[9], offer[10], offer[11], offer[12], offer[13]]
            cv_s_skills = [cv[10], cv[11], cv[12], cv[13], cv[14]]
            cv_age, age_min, age_max = cv[15], offer[14], offer[15]
            cv_languages = [cv[16], cv[17], cv[18]]
            offer_languages = [offer[16], offer[17], offer[18]]
            cv_sm, offer_sm = cv[22], offer[22]
            cv_ea, offer_ea = cv[23], offer[23]
            cv_cert, offer_cert = cv[19], offer[19]

            score[idx, 0] = offer[0]
            score[idx, 1] = cv[0]
            score[idx, 2] = self.educationScore(offer[2], cv[2])
            score[idx, 3] = self.cityScore(offer[3], cv[3], cv[4])
            score[idx, 4] = self.skillScore(offer_skills, cv_skills)
            score[idx, 5] = self.softSkillScore(offer_s_skills, cv_s_skills)
            score[idx, 6] = self.ageScore(cv_age, age_min, age_max)
            score[idx, 7] = self.languageScore(offer_languages, cv_languages)
            score[idx, 8] = self.certificateScore(offer_cert, cv_cert)  # certificate no skill!!!!!
            score[idx, 9] = self.experienceScore(offer[20], cv[20]) if cv[1] == offer[1] else 0
            score[idx, 10] = self.salaryScore(offer[21], cv[21])
            score[idx, 11] = (1 if cv_sm else 0) if offer_sm else 0  # SmartWork max 1, min 0
            score[idx, 12] = (1 if cv_ea else 0) if offer_ea else 0  # Experience_abroad max 1, min 0

        score = DataFrame(data=score,
                          columns=["qId", "kId", "Education", "City", "Skills", "SoftSkills",
                                   "Age", "Language", "Certificates", "Experience", "Salary",
                                   "SmartWork", "Experience_abroad"],
                          dtype=np.float32)

        score = score.astype({"qId": "int", "kId": "int", "SmartWork": "int", "Experience_abroad": "int"})

        features = score.iloc[:, 2:13]

        # Simple sum
        score["score"] = features.sum(axis=1)

        # Weighted sum
        score['w_score'] = features.apply(lambda row: np.dot(row, self.weights), axis=1)

        # Summing in both the random noise
        score["score"] += normal(self.noise[0], self.noise[1], score.shape[0])  # random noise
        score['w_score'] += normal(self.noise[0], self.noise[1], score.shape[0])  # random noise

        # relevance
        intervals, edges = np.histogram(score.sort_values("w_score", ascending=False)["w_score"].to_numpy(),
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

        score["relevance"] = score['w_score'].apply(score2label)

        if output_file is not None:
            score.to_csv(f"../outputs/scores/{output_file}.csv", index=False)

        train, test = train_test_split(score, test_size=self.split_size[0], random_state=self.split_seed)
        train, valid = train_test_split(train, test_size=self.split_size[1], random_state=self.split_seed)

        if output_file is not None:
            train.to_csv(f"../outputs/scores/{output_file}_tr.csv", index=False)
            valid.to_csv(f"../outputs/scores/{output_file}_vl.csv", index=False)
            test.to_csv(f"../outputs/scores/{output_file}_ts.csv", index=False)

        return score
