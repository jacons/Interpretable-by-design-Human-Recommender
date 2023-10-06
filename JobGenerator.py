import json
import random
from itertools import product

import numpy as np
import pandas as pd
from numpy import arange
from numpy.random import choice
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
        self.index = 0

        self.weights = np.array([
            1,  # offer id (always "1" constant)
            1,  # cv id (always "1" constant)

            10,  # Job_relevance
            7,  # Education
            0.5,  # SmartWork
            0.5,  # Experience_abroad
            1,  # City
            8,  # Skills
            6,  # Soft-kills
            2,  # Age
            2,  # Experience
            1,  # Offered_Salary
            0.5,  # Language

        ], dtype=np.float32)
        self.weights[2:] /= self.weights[2:].sum()

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

    def generate_curriculum(self) -> dict:
        jobName = random.choice(self.idx2jobName)
        current_job = self.jobs[jobName]

        # we impose the minimal instruction given the job
        degree = "P2" if current_job["degree"] else "P1"

        """
            ##### Observations #####:
                Ideal salary (is depended of) Experience.
                Experience (is depended of) Age.
                Age (is depended of) Education.
                Education (is depended of) type of work
                
                Language (is depended of) Nationality
        """

        cv = dict(
            Id=self.index,
            Job=jobName,
            Education=self.education.sample(n=1, weights=degree)["Education"].values[0],
            SmartWork=random.randint(0, 1) == 0 if current_job["smart_working"] else False,
            Experience_abroad=random.randint(0, 1) == 0,
            Nationality=self.nationality.sample(n=1, weights="P")["Nationality"].values[0],
            City=self.all_citizen[random.randint(0, self.max_city - 1)],
            Skills=self.skill_gen(current_job["skills"]),
            Softkills=self.skill_gen(current_job["soft_skills"])
        )
        # Min age based on a type of education
        min_age = self.education[self.education["Education"] == cv["Education"]]["Min_age"].values[0]
        min_age = max(min_age, current_job["age"][0])

        cv["Age"] = random.randint(min_age, current_job["age"][1])
        cv["Experience"] = random.randint(0, cv["Age"] - min_age)

        # Retrieve min and max salary
        min_salary, max_salary = current_job["salary"]
        # The min salary for this kind of curriculum is given by
        # min_salary + 7% the salary of each year of experience
        min_salary += int((min_salary * 0.07) * cv["Experience"])
        min_salary = min(min_salary, max_salary)

        cv["Ideal_Salary"] = random.randint(min_salary, max_salary)
        cv["Language"] = self.languages_gen(cv["Nationality"])

        self.index += 1
        return cv

    def generate_jobOffer(self) -> dict:

        jobName = random.choice(self.idx2jobName)
        current_job = self.jobs[jobName]

        """
            ##### Observations #####:
                Ideal salary (is depended of) Experience.
                Experience (is depended of) Age.
                Age (is depended of) Education.
                Education (is depended of) type of work
                
                Language (is depended of) Nationality
        """

        degree = "P2" if current_job["degree"] else "P1"

        offer = dict(
            Id=self.index,
            Job=jobName,
            Education=self.education.sample(n=1, weights=degree)["Education"].values[0],
            SmartWork=random.randint(0, 1) == 0 if current_job["smart_working"] else False,
            Experience_abroad=random.randint(0, 1) == 0,
            Nationality=self.nationality.sample(n=1, weights="P")["Nationality"].values[0],
            City=self.all_citizen[random.randint(0, self.max_city - 1)],
            Skills=self.skill_gen(current_job["skills"]),
            Softkills=self.skill_gen(current_job["soft_skills"]),
        )
        # Min age based on a type of education
        min_age = self.education[self.education["Education"] == offer["Education"]]["Min_age"].values[0]
        min_age = max(min_age, current_job["age"][0])
        max_age = random.randint(min_age, current_job["age"][1])

        offer["Age"] = f"{min_age}-{max_age}"
        offer["Experience"] = random.randint(0, max_age - min_age)

        # The min salary for this kind of curriculum is given by
        # min_salary + 7% the salary of each year of experience
        min_salary, max_salary = current_job["salary"]  # Retrieve min and max salary
        min_salary += int((min_salary * 0.07) * offer["Experience"])
        min_salary = min(min_salary, max_salary)

        offer["Offered_Salary"] = random.randint(min_salary, max_salary)
        offer["Language"] = self.languages_gen(offer["Nationality"])

        self.index += 1
        return offer

    def ScoreFunction(self, offers: DataFrame, cvs: DataFrame) -> DataFrame:

        combinations = list(product(offers.itertuples(index=False), cvs.itertuples(index=False)))
        score = np.zeros((len(combinations), 13), dtype=np.float32)

        for idx, (offer, cv) in enumerate(tqdm(combinations)):
            edu_offer = self.education_ranks.get(offer[2])
            edu_cv = self.education_ranks.get(cv[2])

            cv_skill, offer_skill = cv[7].split(", "), offer[7].split(", ")
            cv_s_skill, offer_s_skill = cv[8].split(", "), offer[8].split(", ")
            age_o = offer[9].split("-")
            cv_languages, offer_languages = cv[12].split(", "), offer[12].split(", ")
            age_min, age_max = int(age_o[0]), int(age_o[1])

            score[idx][0] = offer[0]
            score[idx][1] = cv[0]
            score[idx][2] = 1 if offer[1] == cv[1] else -1  # Job_relevance, max 1, min -1
            score[idx][3] = 1 if edu_cv >= edu_offer else edu_cv - edu_offer  # Education max 1, min -5
            score[idx][4] = (1 if cv[3] else -1) if offer[3] else 0  # SmartWork max 1, min -1
            score[idx][5] = 1 if cv[4] == offer[4] else -1  # Experience_abroad max 1, min -1
            score[idx][6] = self.distance_city(offer[6], cv[6])  # City max 1, min 0
            score[idx][7] = self.find_similar(cv_skill, offer_skill)  # Skills
            score[idx][8] = self.simple_intersection(cv_s_skill, offer_s_skill)  # Soft-kills max 1, min 0
            score[idx][9] = self.score_piecewise_f(cv[9], age_min, age_max, 7)  # Age max 1, min -inf

            # Experience max 1, min 0
            score[idx][10] = 1 if offer[10] <= cv[10] else max((cv[10] - offer[10]) / 6 + 1, 0)
            # Salary max 1, min -inf
            score[idx][11] = 1 if cv[11] <= offer[11] else max((-cv[11] + offer[11]) / 300 + 1, 0)

            score[idx][12] = self.simple_intersection(cv_languages, offer_languages)  # Language max 1, min 0

        score = pd.DataFrame(data=score * self.weights,
                             columns=["id_offer", "id_cv", "Job relevance", "Education", "SmartWork",
                                      "Experience_abroad", "City", "Skills", "Soft-skills",
                                      "Age", "Experience", "Salary", "Language"],
                             dtype=np.float32)

        score["score"] = score.iloc[:, 2:13].sum(axis=1)
        score["score"] += np.random.normal(0, 0.3, score.shape[0])  # random noise
        # 0.3
        score.to_csv("outputs/scores.csv", index=False)
        return score
