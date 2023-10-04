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
        In also can match them to generate the score(or label) (used to supervised task).
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
            {'Rank': 'int', 'Education': 'string', 'P1': 'float', 'P2': 'float'})

        # dictionary that map the id to a job name
        self.idx2jobName = {idx: k for idx, k in enumerate(self.jobs.keys())}
        self.max_city = len(self.all_citizen)

        self.skill2job = self.generate_skill2job()
        self.index = 0

        self.weights = np.array([
            1,  # offer id (always "1" constant)
            1,  # cv id (always "1" constant)

            10,  # Job_relevance
            8,  # Education
            2,  # Age
            3,  # Experience
            10,  # Skills
            6,  # Soft-kills
            2,  # Offered_Salary
            1,  # City
            0.5,  # SmartWork
            0.7,  # Experience_abroad
            1,  # Language
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
    def score_age(ageO: str, ageC: int):
        age = ageO.split()
        if int(age[0]) < ageC < int(age[2]):
            return 1
        else:
            return -min(np.abs(int(age[0]) - ageC), np.abs(int(age[2]) - ageC))

    @staticmethod
    def simple_intersection(a: list, b: list) -> float:
        return len(set(a) & set(b)) / len(b)

    def distance_city(self, cityA: str, cityB: str):
        return self.distance.loc[(cityA, cityB)].values[0]

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
                    score += 0.5
                    break
        return score

    def generate_curriculum(self) -> dict:
        jobName = self.idx2jobName[random.randint(0, len(self.idx2jobName) - 1)]
        current_job = self.jobs[jobName]

        degree = "P2" if current_job["degree"] else "P1"
        mu_salary = sum(current_job["salary"]) / 2
        cv = dict(
            Id=self.index,
            Job=jobName,
            Education=self.education.sample(n=1, weights=degree)["Education"].values[0],
            Age=random.randint(20, 70),
            Experience=random.randint(*current_job["experience"]),
            Skills=self.skill_gen(current_job["skills"]),
            Softkills=self.skill_gen(current_job["soft_skills"]),
            Ideal_Salary=int(random.gauss(mu_salary, 10)),
            City=self.all_citizen[random.randint(0, self.max_city - 1)],
            Nationality=self.nationality.sample(n=1, weights="P")["Nationality"].values[0],
            SmartWork=current_job["smart_working"],
            Experience_abroad=random.randint(0, 1) == 0,
        )
        cv["Language"] = self.languages_gen(cv["Nationality"])

        self.index += 1
        return cv

    def generate_jobOffer(self) -> dict:

        jobName = self.idx2jobName[random.randint(0, len(self.idx2jobName) - 1)]
        current_job = self.jobs[jobName]

        age1, age2 = random.randint(*current_job["age"]), random.randint(*current_job["age"])
        if age1 > age2:
            age1, age2 = age2, age1

        degree = "P2" if current_job["degree"] else "P1"
        mu_salary = sum(current_job["salary"]) / 2
        offer = dict(
            Id=self.index,
            Job=jobName,
            Education=self.education.sample(n=1, weights=degree)["Education"].values[0],
            Age=str(age1) + " - " + str(age2),
            Experience=random.randint(*current_job["experience"]),
            Skills=self.skill_gen(current_job["skills"]),
            Softkills=self.skill_gen(current_job["soft_skills"]),
            Offered_Salary=int(random.gauss(mu_salary, 10)),
            City=self.all_citizen[random.randint(0, self.max_city - 1)],
            Nationality=self.nationality.sample(n=1, weights="P")["Nationality"].values[0],
            SmartWork=current_job["smart_working"],
            Experience_abroad=random.randint(0, 1) == 0
        )
        offer["Language"] = self.languages_gen(offer["Nationality"])

        self.index += 1
        return offer

    def ScoreFunction(self, offers: DataFrame, cvs: DataFrame) -> DataFrame:

        combinations = list(product(offers.itertuples(index=False), cvs.itertuples(index=False)))
        matrix_score = np.zeros((len(combinations), 13), dtype=np.float32)

        for idx, (offer, cv) in enumerate(tqdm(combinations)):
            edu_offer = self.education[self.education["Education"] == offer[2]]["Rank"].values[0]
            edu_cv = self.education[self.education["Education"] == cv[2]]["Rank"].values[0]

            cv_languages, offer_languages = cv[12].split(", "), offer[12].split(", ")
            cv_skill, offer_skill = cv[5].split(", "), offer[5].split(", ")
            cv_s_skill, offer_s_skill = cv[6].split(", "), offer[6].split(", ")

            matrix_score[idx][0] = offer[0]
            matrix_score[idx][1] = cv[0]
            matrix_score[idx][2] = 1 if offer[1] == cv[1] else -1  # Job_relevance, max 1, min -1
            matrix_score[idx][3] = 1 if edu_cv >= edu_offer else edu_cv - edu_offer  # Education max 1, min -5
            matrix_score[idx][4] = self.score_age(offer[3], cv[3])  # Age max 1, min -inf
            matrix_score[idx][5] = 1 if cv[4] >= offer[4] else cv[4] - offer[4]  # Experience max 1, min -inf
            matrix_score[idx][6] = self.find_similar(cv_skill, offer_skill)  # Skills
            matrix_score[idx][7] = self.simple_intersection(cv_s_skill, offer_s_skill)  # Soft-kills max 1, min 0
            matrix_score[idx][8] = 1 if offer[7] >= cv[7] else offer[7] - cv[7]  # Offered_Salary max 1, min -inf
            matrix_score[idx][9] = self.distance_city(offer[8], cv[8])  # City max 1, min 0
            matrix_score[idx][10] = 1 if cv[10] == offer[10] else -1  # SmartWork max 1, min -1
            matrix_score[idx][11] = 1 if cv[11] == offer[11] else -1  # Experience_abroad max 1, min -1
            matrix_score[idx][12] = self.simple_intersection(cv_languages, offer_languages)  # Language max 1, min 0

        matrix_score = pd.DataFrame(data=matrix_score * self.weights,
                                    columns=["id_offer", "id_cv", "Job relevance", "Education", "Age", "Experience",
                                             "Skills", "Soft-skills", "Salary", "City", "SmartWork",
                                             "Experience_abroad", "Language"],
                                    dtype=np.float32)
        matrix_score["score"] = matrix_score.iloc[:, 2:13].sum(axis=1)

        matrix_score.to_csv("outputs/scores.csv", index=False)
        return matrix_score
