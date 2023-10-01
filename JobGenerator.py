import json
import random

import pandas as pd
from numpy import arange
from numpy.random import choice


class JobGenerator:
    def __init__(self, jobs_lib: str,
                 citizen: str,
                 nationality: str,
                 education: str):
        with open(jobs_lib, "r") as f:
            self.jobs = json.load(f)

        with open(citizen, "r") as f:
            self.all_citizen = f.read().split("\n")

        self.nationality = pd.read_csv(nationality)
        self.nationality["P"] = self.nationality["P"].astype(float)

        self.education = pd.read_csv(education)
        self.education["P1"] = self.education["P1"].astype(float)
        self.education["P2"] = self.education["P2"].astype(float)

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
            Age=random.randint(0, 70),
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

