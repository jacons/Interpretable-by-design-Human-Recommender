from itertools import product
from typing import Tuple

import numpy as np
from numpy.random import normal
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Generator.JobGenerator import JobGenerator
from Generator.JobGraph import JobGraph


class MatchingScore:
    def __init__(self,
                 jobGenerator: JobGenerator,
                 cities_dist: str,
                 bins: int,
                 weight: np.ndarray,
                 noise: Tuple[float],
                 split_size: Tuple[float],
                 split_seed: int):

        self.distance = read_csv(cities_dist, index_col=[0, 1], skipinitialspace=True)
        self.max_distance = self.distance["D"].max()

        self.education = {i[1]: i[0] for i in jobGenerator.education.itertuples()}
        self.lvl2value = {v: i for i, v in enumerate(["A1", "A2", "B1", "B2", "C1", "C2"])}
        self.education["-"] = "-"
        self.len_ed_rank = len(self.education)
        return
        # self.bins = bins  # Number of relevances
        # self.noise = noise # mean and stddev
        # self.split_size = split_size
        # self.split_seed = split_seed

        # self.job_graph = jobGenerator.job_graph

        # self.weights = self.normalize_weights(weight)

    def educationScore(self, offer_ess: str, offer_op: str, cv: str) -> int:
        # max 1,25 min 0
        cv = self.education[cv]  # level of candidate's education
        offer_ess = self.education[offer_ess]  # essential education
        offer_op = self.education[offer_op]  # optional education

        score = 1 if offer_ess <= cv else 0
        bonus = 0 if offer_op == "-" else 0.25 if offer_op <= cv else 0

        return score + bonus

    def cityScore(self, cityA: str, cityB: str, range_: int) -> float:
        # max 1 min 0
        if cityA == cityB:
            return 1

        s_cities = sorted([cityA, cityB])
        dist = self.distance.loc[(s_cities[0], s_cities[1])].values[0]

        return 1 if dist < range_ else 1 - (dist - range_) / self.max_distance

    @staticmethod
    def ageScore(cv: int, v_min: int, v_max: int) -> float:
        # max 1 min 0
        return 1 if int(v_min <= cv <= v_max) else 0

    @staticmethod
    def experienceScore(offer_ess: str, offer_op: bool, cv: int) -> float:
        # max 1,25 min 0
        score, bonus = 0, 0
        if offer_ess != "-":
            score += 1 if int(offer_ess) <= cv else 0
            bonus += 0.25 if int(offer_ess) <= cv else 0 if offer_op else 0

        return score + bonus

    def languageScore(self, essential: list[tuple], cv: list[tuple], optional: tuple) -> float:

        score, bonus = 0, 0
        for a, b in product(essential, cv):
            if a[0] == b[0]:
                lvl_of = self.lvl2value[a[1]]
                lvl_cv = self.lvl2value[b[1]]
                score += 1 if lvl_of <= lvl_cv else 1 / (lvl_of - lvl_cv)

        for lang, lvl in cv:
            if lang == optional[0]:
                lvl_of = self.lvl2value[optional[1]]
                lvl_cv = self.lvl2value[lvl]
                bonus += 0.25 if lvl_of <= lvl_cv else 0.25 / (lvl_of - lvl_cv)

        return score / len(essential) + bonus

    def skillScore(self, essential: list, optional: list, cv: list):
        pass

    @staticmethod
    def remove_null(a: list, b: list):
        list_ = []
        for lang, lvl in zip(a, b):
            if lang != "-":
                list_.append((lang, lvl))
        return list_

    @staticmethod
    def filter(list_: list):
        return [item for item in list_ if item != "-"]

    def scoreFunction(self, offers: DataFrame, curricula: DataFrame, output_file: str = None):
        for qId, offer in offers.iterrows():
            curricula = curricula[curricula["qId"] == qId]

            print(offer)
            for kId, cv in curricula.iterrows():
                print(cv)

                cv_lang = self.remove_null([cv["Language0"], cv["Language1"], cv["Language2"]],
                                           [cv["Language_level0"], cv["Language_level1"], cv["Language_level2"]])
                of_lang = self.remove_null([offer["Language_essential0"], offer["Language_essential1"]],
                                           [offer["Language_level0"], offer["Language_level1"]])

                of_comp_ess = self.filter([offer[f"Competence_essential{i}"] for i in range(4)])
                of_comp_opt = self.filter([offer[f"Competence_optional{i}"] for i in range(3)])
                of_know_ess = self.filter([offer[f"Knowledge_essential{i}"] for i in range(4)])
                of_know_opt = self.filter([offer[f"Knowledge_optional{i}"] for i in range(3)])
                cv_comp = self.filter([cv[f"Competences{i}"] for i in range(6)])
                cv_know = self.filter([cv[f"Knowledge{i}"] for i in range(6)])

                fitness_edu = self.educationScore(offer["Edu_essential"],
                                                  offer["Edu_optional"],
                                                  cv["Education"])
                fitness_city = self.cityScore(offer["City"], cv["City"], cv["JobRange"])
                fitness_age = self.ageScore(cv["Age"], offer["AgeMin"], offer["AgeMax"])
                fitness_exp = self.experienceScore(offer["Experience_essential"],
                                                   offer["Experience_optional"],
                                                   cv["Experience"])
                fitness_lang = self.languageScore(of_lang, cv_lang,
                                                  (offer["Language_optional0"], offer["Language_level2"]))

            break


"""
    @staticmethod
    def normalize_weights(weights: np.ndarray):
        return weights / weights.sum()



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

            score[idx, 4] = self.skillScore(offer_skills, cv_skills)

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
"""
