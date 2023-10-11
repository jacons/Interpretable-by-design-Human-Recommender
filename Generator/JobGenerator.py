import json
import random
from itertools import product

import numpy as np
import pandas as pd
from numpy import arange
from numpy.random import choice, normal
from pandas import DataFrame
from tqdm import tqdm


class JobGenerator:
    def __init__(self, jobs_lib: str, citizen: str, nationality: str, education: str, citizen_dist: str):
        """
        JobGenerator is a tool that allows us to generate synthetic data about the "curriculums" and "jobs offer".
        In also can match them to generate the score(or label) (supervised task).
        :param jobs_lib: Predefined jobs
        :param citizen: Subsample of all citizens
        :param nationality: Subsample of all nationality
        :param education: Education hierarchy
        :param citizen_dist: Distances among the previous citizens
        """

        with open(jobs_lib, "r") as f:
            self.jobs = json.load(f)

        with open(citizen, "r") as f:
            self.all_citizen = f.read().split("\n")

        self.nationality = pd.read_csv(nationality).astype(
            {'Nationality': 'string', 'P': 'float'})

        self.distance = pd.read_csv(citizen_dist, index_col=[0, 1], skipinitialspace=True)

        self.education = pd.read_csv(education).astype(
            {'Rank': 'int', 'Education': 'string', 'P1': 'float',
             'P2': 'float', 'Min_age': 'int'})

        self.education_ranks = dict(zip(self.education["Education"], self.education["Rank"]))

        # dictionary that map the id to a job name
        self.idx2jobName = {idx: k for idx, k in enumerate(self.jobs.keys())}
        self.max_city = len(self.all_citizen)

        self.skill2job = self.generate_skill2job()

        self.weights = np.array([
            # 9,  # Job_relevance
            7,  # Education
            1,  # SmartWork
            1,  # Experience_abroad
            1,  # City
            15,  # Skills
            8,  # Softskills
            2,  # Age
            2,  # Experience
            1,  # Offered_Salary
            1,  # Language

        ], dtype=np.float32)
        self.weights /= self.weights.sum()

    def generate_skill2job(self) -> dict:
        skill2job = {}

        for k, v in self.jobs.items():
            for s in v["skills"]:
                if s not in skill2job:
                    skill2job[s] = [k]
                else:
                    if k not in skill2job[s]:
                        skill2job[s] += [k]
        return skill2job

    @staticmethod
    def skill_gen(list_skill: list):
        max_skill = len(list_skill)
        n_skill = min(random.randint(2, 5), max_skill)
        r_idx_skill = choice(arange(0, max_skill), size=n_skill, replace=False)
        str_skill = ", ".join(list(map(lambda x: list_skill[x], r_idx_skill)))
        return str_skill

    def languages_gen(self, nationality: str) -> str:
        lang_list = [nationality]

        other_languages = self.nationality[self.nationality["Nationality"] != nationality]
        for i in range(random.randint(0, 2)):
            lang_list.append(other_languages.sample(n=1, weights="P")["Nationality"].values[0])

        return ", ".join(lang_list)

    @staticmethod
    def score_piecewise_f(value: int, v_min: int, v_max: int, slope: int):
        # Piecewise function to score. Max 1 Min 0

        if int(v_min <= value <= v_max):
            return 1
        if value > v_max:
            return max((-value + v_max) / slope + 1, 0)
        return max((value - v_min) / slope + 1, 0)

    @staticmethod
    def simple_intersection(a: list, b: list) -> float:
        return len(set(a) & set(b)) / len(b)

    def distance_city(self, cityA: str, cityB: str):
        # same city maximal point
        return 1 if cityA == cityB else self.distance.loc[(cityA, cityB)].values[0]

    def find_similar(self, cv_skills: list, offer_skills: list) -> float:

        intersect = list(set(cv_skills) & set(offer_skills))
        score = len(intersect) / len(offer_skills)  # "1" perfect, "0" bad

        for i_ in intersect:
            offer_skills.remove(i_)
            cv_skills.remove(i_)

        # Retrieve all jobs where there is at least one skill
        similar_jobs = []
        for skill_ in offer_skills:
            similar_jobs += self.skill2job[skill_]

        for skill_ in cv_skills:
            for job_ in similar_jobs:
                if skill_ in self.jobs[job_]["skills"]:
                    plus = (1 - score) / 2
                    score += plus
                    break
        return score

    def get_curricula(self, size: int = 1, path: str = None) -> DataFrame:

        curricula = [self.__curriculum() for _ in range(size)]

        curricula = pd.DataFrame(curricula).astype(
            dtype={"SmartWork": "int", "Experience_abroad": "int"})

        if path is not None:
            curricula.to_csv(path, index_label="kId")

        return curricula

    def get_offers(self, size: int = 1, path: str = None) -> DataFrame:

        offers = [self.__jobOffer() for _ in range(size)]

        offers = pd.DataFrame(offers).astype(
            dtype={"SmartWork": "int", "Experience_abroad": "int"})

        if path is not None:
            offers.to_csv(path, index_label="qId")

        return offers

    def __curriculum(self) -> dict:
        """
            ##### Observations #####:
                Ideal salary (is depended on) Experience.
                Experience (is depended on) Age.
                Age (is depended on) Education.
                Education (is depended on) type of work
                Language (is depended on) Nationality
        """

        jobName = random.choice(self.idx2jobName)
        current_job = self.jobs[jobName]

        # we impose the minimal instruction given the job
        degree = "P2" if current_job["degree"] else "P1"

        nationality = self.nationality.sample(n=1, weights="P")["Nationality"].values[0]

        cv = dict(
            # Job=jobName,
            Education=self.education.sample(n=1, weights=degree)["Education"].values[0],  # 1
            SmartWork=random.randint(0, 1) == 0 if current_job["smart_working"] else False,  # 2
            Experience_abroad=random.randint(0, 1) == 0,  # 3
            City=self.all_citizen[random.randint(0, self.max_city - 1)],  # 4
            Skills=self.skill_gen(current_job["skills"]),  # 5
            Softskills=self.skill_gen(current_job["soft_skills"])  # 6
        )
        # Min age based on a type of education
        min_age = self.education[self.education["Education"] == cv["Education"]]["Min_age"].values[0]
        min_age = max(min_age, current_job["age"][0])

        cv["Age"] = random.randint(min_age, current_job["age"][1])  # 7
        cv["Experience"] = random.randint(0, cv["Age"] - min_age)  # 8

        # Retrieve min and max salary
        min_salary, max_salary = current_job["salary"]
        # The min salary for this kind of curriculum is given by
        # min_salary + 7% the salary of each year of experience
        min_salary += int((min_salary * 0.07) * cv["Experience"])
        min_salary = min(min_salary, max_salary)

        cv["Ideal_Salary"] = random.randint(min_salary, max_salary)  # 9
        cv["Language"] = self.languages_gen(nationality)  # 10

        return cv

    def __jobOffer(self) -> dict:
        """
            ##### Observations #####:
                Ideal salary (is depended on) Experience.
                Experience (is depended on) Age.
                Age (is depended on) Education.
                Education (is depended on) type of work
                
                Language (is depended on) Nationality
        """

        jobName = random.choice(self.idx2jobName)
        current_job = self.jobs[jobName]

        degree = "P2" if current_job["degree"] else "P1"

        nationality = self.nationality.sample(n=1, weights="P")["Nationality"].values[0]

        offer = dict(
            Job=jobName,  # 1
            Education=self.education.sample(n=1, weights=degree)["Education"].values[0],  # 2
            SmartWork=random.randint(0, 1) == 0 if current_job["smart_working"] else False,  # 3
            Experience_abroad=random.randint(0, 1) == 0,  # 4
            City=self.all_citizen[random.randint(0, self.max_city - 1)],  # 5
            Skills=self.skill_gen(current_job["skills"]),  # 6
            Softskills=self.skill_gen(current_job["soft_skills"]),  # 7
        )
        # Min age based on a type of education
        min_age = self.education[self.education["Education"] == offer["Education"]]["Min_age"].values[0]
        min_age = max(min_age, current_job["age"][0])
        max_age = random.randint(min_age, current_job["age"][1])

        offer["Age"] = f"{min_age}-{max_age}"  # 8
        offer["Experience"] = random.randint(0, max_age - min_age)  # 9

        # The min salary for this kind of curriculum is given by
        # min_salary + 7% the salary of each year of experience
        min_salary, max_salary = current_job["salary"]  # Retrieve min and max salary
        min_salary += int((min_salary * 0.07) * offer["Experience"])
        min_salary = min(min_salary, max_salary)

        offer["Offered_Salary"] = random.randint(min_salary, max_salary)  # 10
        offer["Language"] = self.languages_gen(nationality)  # 11

        return offer

    def ScoreFunction(self, offers: DataFrame, cvs: DataFrame, path: str = None) -> DataFrame:

        combinations = list(product(offers.itertuples(), cvs.itertuples()))
        score = np.zeros((len(combinations), 12), dtype=np.float32)

        for idx, (offer, cv) in enumerate(tqdm(combinations)):
            edu_cv, edu_offer = self.education_ranks.get(cv[1]), self.education_ranks.get(offer[2])

            cv_skill, offer_skill = cv[5].split(", "), offer[6].split(", ")
            cv_s_skill, offer_s_skill = cv[6].split(", "), offer[7].split(", ")
            age_o = offer[8].split("-")
            cv_languages, offer_languages = cv[10].split(", "), offer[11].split(", ")
            age_min, age_max = int(age_o[0]), int(age_o[1])

            score[idx][0] = offer[0]
            score[idx][1] = cv[0]

            # score[idx][2] = 1 if offer[1] == cv[1] else 0  # Job_relevance, max 1, min 0
            score[idx][2] = 1 if edu_cv >= edu_offer else 1 - (edu_offer - edu_cv) / 5  # Education max 1, min 0
            score[idx][3] = (1 if cv[2] else 0) if offer[3] else 0  # SmartWork max 1, min 0
            score[idx][4] = 1 if cv[3] == offer[4] else 0  # Experience_abroad max 1, min 0
            score[idx][5] = self.distance_city(cv[4], offer[5])  # City max 1, min 0
            score[idx][6] = self.find_similar(cv_skill, offer_skill)  # Skills max 1, min 0
            score[idx][7] = self.simple_intersection(cv_s_skill, offer_s_skill)  # Soft-kills max 1, min 0
            score[idx][8] = self.score_piecewise_f(cv[7], age_min, age_max, 7)  # Age max 1, min 0

            # Experience max 1, min 0
            score[idx][9] = 1 if cv[8] >= offer[9] else max((cv[8] - offer[9]) / 6 + 1, 0)
            # Salary max 1, min 0
            score[idx][10] = 1 if cv[9] <= offer[10] else max((offer[10] - cv[9]) / 300 + 1, 0)

            score[idx][11] = self.simple_intersection(cv_languages, offer_languages)  # Language max 1, min 0

        score = pd.DataFrame(data=score,
                             columns=["qId", "kId", "Education", "SmartWork", "Experience_abroad",
                                      "City", "Skills", "Softskills", "Age", "Experience", "Salary", "Language"],
                             dtype=np.float32)
        score = score.astype({"qId": "int", "kId": "int", "SmartWork": "int",
                              "Experience_abroad": "int", "Salary":"int"})

        # ---------------- Score generation ----------------
        features = score.iloc[:, 2:12]

        # Simple sum
        score["score"] = features.sum(axis=1)

        # Weighted sum
        score['w_score'] = features.apply(lambda row: np.dot(row, self.weights), axis=1)

        # Summing in both the random noise
        score["score"] += normal(0, 0.2, score.shape[0])  # random noise
        score['w_score'] += normal(0, 0.2, score.shape[0])  # random noise

        # labels
        intervals, edges = np.histogram(score.sort_values("w_score", ascending=False)["w_score"].to_numpy(),
                                        bins=10)
        score2inter = {i: (edges[i], edges[i + 1]) for i in range(len(intervals))}

        def score2label(score_value: float) -> int:
            if score_value <= score2inter[0][0]:
                return 0
            for i, (v_min, v_max) in score2inter.items():
                if v_min <= score_value < v_max:
                    return i
            if score_value >= score2inter[9][1]:
                return 9

        score["labels"] = score['w_score'].apply(score2label)

        # ---------------- Score generation ----------------

        if path is not None:
            score.to_csv(path, index=False)
        return score
