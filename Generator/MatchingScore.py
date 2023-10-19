from itertools import product

import numpy as np
from numpy.random import normal
from pandas import read_csv, DataFrame
from tqdm import tqdm

from Generator.JobGenerator import JobGenerator
from Generator.SkillGraph import SkillGraph


class MatchingScore:
    def __init__(self, citizen_dist: str, jobGenerator: JobGenerator):
        self.skillGraph = SkillGraph(jobGenerator.jobs)

        self.education_ranks = dict(zip(jobGenerator.education["Education"], jobGenerator.education["Rank"]))
        self.len_ed_rank = len(self.education_ranks)

        self.distance = read_csv(citizen_dist, index_col=[0, 1], skipinitialspace=True)
        self.max_distance = self.distance["D"].max()

        self.lvl2value = {t[1]: t[0] for t in jobGenerator.languages_level.itertuples()}

        self.weights = np.array([
            8,  # Education
            2,  # City
            15,  # Skills
            5,  # SoftSkills
            2,  # Age
            6,  # Language
            3,  # Certificates
            2,  # Experience
            1,  # Offered_Salary
            1,  # SmartWork
            1,  # Experience abroad

        ], dtype=np.float32)
        self.weights /= self.weights.sum()

    def educationScore(self, offer: str, cv: str) -> int:
        edu_cv = self.education_ranks.get(cv)
        edu_offer = self.education_ranks.get(offer)

        # Education max 1, min 0
        return 1 if edu_cv >= edu_offer else 1 - (edu_offer - edu_cv) / self.len_ed_rank

    def cityScore(self, cityA: str, cityB: str, range_: int) -> float:

        if cityA == cityB:
            return 1

        s_citizen = sorted([cityA, cityB])
        dist = self.distance.loc[(s_citizen[0], s_citizen[1])].values[0]

        return 1 if dist < range_ else 1 - (dist - range_) / self.max_distance

    def skillScore(self, offer: list, cv: list) -> float:

        offer = set([x for x in offer if x != "-"])
        cv = set([x for x in cv if x != "-"])

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
                    paths.append(1 / n_hops)
                plus += min(paths)
            plus /= len(offer)

        return score + plus

    @staticmethod
    def softSkillScore(offer: list, cv: list) -> float:

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
        return 1 if cv >= offer else max((cv - offer) / 6 + 1, 0)

    @staticmethod
    def salaryScore(offer: int, cv: int) -> float:
        # Salary max 1, min 0
        return 1 if cv <= offer else max((offer - cv) / 300 + 1, 0)

    def scoreFunction(self, offers: DataFrame, cvs: DataFrame, path: str = None):
        combinations = list(product(offers.itertuples(), cvs.itertuples()))
        score = np.zeros((len(combinations), 13), dtype=np.float32)

        for idx, (offer, cv) in enumerate(tqdm(combinations)):
            offer_skills = [offer[4], offer[5], offer[6], offer[7], offer[8]]
            cv_skills = [cv[4], cv[5], cv[6], cv[7], cv[8]]
            offer_s_skills = [offer[9], offer[10], offer[11], offer[12], offer[13]]
            cv_s_skills = [cv[9], cv[10], cv[11], cv[12], cv[13]]
            cv_age, age_min, age_max = cv[14], offer[14], offer[15]
            cv_languages = [cv[15], cv[16], cv[17]]
            offer_languages = [offer[16], offer[17], offer[18]]
            cv_sm, offer_sm = cv[21], offer[22]
            cv_ea, offer_ea = cv[22], offer[23]
            cv_cert, offer_cert = cv[18], offer[19]

            score[idx, 0] = offer[0]
            score[idx, 1] = cv[0]
            score[idx, 2] = self.educationScore(offer[2], cv[1])
            score[idx, 3] = self.cityScore(offer[3], cv[2], cv[3])
            score[idx, 4] = self.skillScore(offer_skills, cv_skills)
            score[idx, 5] = self.softSkillScore(offer_s_skills, cv_s_skills)
            score[idx, 6] = self.ageScore(cv_age, age_min, age_max)
            score[idx, 7] = self.languageScore(offer_languages, cv_languages)
            score[idx, 8] = self.softSkillScore(offer_cert, cv_cert)
            score[idx, 9] = self.experienceScore(offer[20], cv[19])
            score[idx, 10] = self.salaryScore(offer[21], cv[20])
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
        score["score"] += normal(0, 0.2, score.shape[0])  # random noise
        score['w_score'] += normal(0, 0.2, score.shape[0])  # random noise

        # labels
        intervals, edges = np.histogram(score.sort_values("w_score", ascending=False)["w_score"].to_numpy(), bins=6)
        score2inter = {i: (edges[i], edges[i + 1]) for i in range(len(intervals))}

        def score2label(score_value: float) -> int:
            if score_value <= score2inter[0][0]:
                return 0
            for i, (v_min, v_max) in score2inter.items():
                if v_min <= score_value < v_max:
                    return i
            if score_value >= score2inter[5][1]:
                return 5

        score["labels"] = score['w_score'].apply(score2label)
        if path is not None:
            score.to_csv(path, index=False)

        return score
