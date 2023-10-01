import json
import random
from itertools import product

import numpy as np
import pandas as pd
from numpy import arange
from numpy.random import choice
from pandas import DataFrame


class JobGenerator:
    def __init__(self, jobs_lib: str,
                 citizen: str,
                 nationality: str,
                 education: str,
                 citizen_dist: str):

        with open(jobs_lib, "r") as f:
            self.jobs = json.load(f)

        with open(citizen, "r") as f:
            self.all_citizen = f.read().split("\n")

        self.nationality = pd.read_csv(nationality)
        self.nationality["P"] = self.nationality["P"].astype(float)

        self.education = pd.read_csv(education)
        self.education["P1"] = self.education["P1"].astype(float)
        self.education["P2"] = self.education["P2"].astype(float)

        self.distance = pd.read_csv(citizen_dist, names=["CityA", "CityB", "Distance"])

        self.idx2jobName = {idx: k for idx, k in enumerate(self.jobs.keys())}
        self.max_city = len(self.all_citizen)

        self.index = 0

    @staticmethod
    def skill_gen(list_skill: list):
        max_skill = len(list_skill)
        n_skill = min(random.randint(2, 5), max_skill)
        r_idx_skill = choice(arange(0, max_skill), size=n_skill, replace=False)
        str_skill = ", ".join(list(map(lambda x: list_skill[x], r_idx_skill)))
        return str_skill

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
            Softskills=self.skill_gen(current_job["soft_skills"]),
            Ideal_Salary=int(random.gauss(mu_salary, 10)),
            City=self.all_citizen[random.randint(0, self.max_city - 1)],
            Nationality=self.nationality.sample(n=1, weights="P")["Nationality"].values[0],
            SmartWork=current_job["smart_working"],
            Experience_abroad=random.randint(0, 1) == 0
        )

        str_lang = cv["Nationality"]
        for i in range(random.randint(0, 2)):
            str_lang += ", " + self.nationality.sample(n=1, weights="P")["Nationality"].values[0]

        cv["Language"] = str_lang

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
            Softskills=self.skill_gen(current_job["soft_skills"]),
            Offered_Salary=int(random.gauss(mu_salary, 10)),
            City=self.all_citizen[random.randint(0, self.max_city - 1)],
            Nationality=self.nationality.sample(n=1, weights="P")["Nationality"].values[0],
            SmartWork=current_job["smart_working"],
            Experience_abroad=random.randint(0, 1) == 0
        )

        str_lang = offer["Nationality"]
        for i in range(random.randint(0, 2)):
            str_lang += ", " + self.nationality.sample(n=1, weights="P")["Nationality"].values[0]

        offer["Language"] = str_lang

        self.index += 1
        return offer

    @staticmethod
    def score_age(ageO: str, ageC: int):
        age = ageO.split()
        if int(age[0]) < ageC < int(age[2]):
            return 1
        else:
            return -max(np.abs(int(age[0]) - ageC), np.abs(int(age[2]) - ageC))

    def distance_city(self, cityA: str, cityB: str):
        return self.distance[(self.distance["CityA"] == cityA) &
                             (self.distance["CityB"] == cityB)]["Distance"].values[0]

    def ScoreFunction(self, offers: DataFrame, cvs: DataFrame):
        combinations = list(product(offers.itertuples(index=False), cvs.itertuples(index=False)))
        for offer, cv in combinations:

            edu_offer = self.education[self.education["Education"] == offer[2]]["Rank"].values[0]
            edu_cv = self.education[self.education["Education"] == cv[2]]["Rank"].values[0]

            cv_languages, offer_languages = cv[12].split(", "), offer[12].split(", ")
            cv_skill, offer_skill = cv[5].split(", "), offer[5].split(", ")
            cv_s_skill, offer_s_skill = cv[6].split(", "), offer[6].split(", ")

            score = dict(
                Job_relevance=1 if offer[1] == cv[1] else -10,
                Education=-np.abs(edu_cv - edu_offer),
                Age=self.score_age(offer[3], cv[3]),
                Experience=1 if cv[4] >= offer[4] else -offer[4] - cv[4],
                Skills=len(list(set(cv_skill) & set(offer_skill))) / len(offer_skill),
                Softskills=len(list(set(cv_s_skill) & set(offer_s_skill))) / len(offer_s_skill),
                Offered_Salary=-np.abs(offer[7] - cv[7]),
                City=self.distance_city(offer[8], cv[8]),
                SmartWork=1 if cv[10] == offer[10] else -1,
                Experience_abroad=1 if cv[11] == offer[11] else -1,
                Language=len(list(set(cv_languages) & set(offer_languages))) / len(offer_languages)
            )
            features = np.array(list(score.values()))
            print(features)
            break
        return
