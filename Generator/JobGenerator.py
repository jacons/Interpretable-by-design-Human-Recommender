import json
import random
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import arange
from numpy.random import choice
from pandas import DataFrame, read_csv
from tqdm import tqdm


class JobGenerator:
    def __init__(self,
                 jobs_lib: str,
                 cities: str,
                 nationality: str,
                 education: str,
                 language_level: str,
                 lang_level_dist:Tuple[float],
                 certificates_dist:Tuple[float]):
        """
        JobGenerator is a tool that allows us to generate synthetic data about the "curriculums" and "jobs offer".
        In also can match them to generate the score(or label) (supervised task).
        :param jobs_lib: Predefined jobs
        :param cities: Subsample of all citizens
        :param nationality: Subsample of all nationality
        :param education: Education hierarchy
        """

        self.certificates_dist = certificates_dist
        self.lang_level_dist = lang_level_dist
        with open(jobs_lib, "r") as f:
            self.jobs = json.load(f)

        with open(cities, "r") as f:
            self.all_cities = f.read().split("\n")

        self.languages = read_csv(nationality).astype(
            {'Languages': 'string', 'P': 'float'})

        self.languages_level = read_csv(language_level).astype(
            {'Level': 'string', 'P': 'float'})

        self.education = read_csv(education).astype(
            {'Rank': 'int', 'Education': 'string', 'P1': 'float',
             'P2': 'float', 'Min_age': 'int'})

        # dictionary that map the id to a job name
        self.idx2jobName = {idx: k for idx, k in enumerate(self.jobs.keys())}
        self.max_city = len(self.all_cities)

    @staticmethod
    def generate_skills(all_skill: list, min_: int = 1, max_: int = 5) -> list[str]:
        max_skill = len(all_skill)
        n_skill = min(random.randint(min_, max_), max_skill)
        r_idx_skill = choice(arange(0, max_skill), size=n_skill, replace=False)
        skills = list(map(lambda x: all_skill[x], r_idx_skill))

        if n_skill < max_:
            skills.extend(["-" for _ in range(n_skill, max_)])
        return skills

    def generate_languages(self, nationality: str, min_: int = 0, max_: int = 2) -> list[str]:

        lang_list = [nationality + " - C2"]
        other_l = self.languages.drop(self.languages[self.languages["Languages"] == nationality].index)

        n_languages = random.choices(arange(min_, max_ + 1), k=1, weights=self.lang_level_dist)[0]
        for i in range(n_languages):
            sampled_language = other_l.sample(n=1, weights="P")["Languages"].values[0]
            sampled_level = self.languages_level.sample(n=1, weights="P")["Level"].values[0]
            other_l = other_l.drop(other_l[other_l["Languages"] == nationality].index)

            lang_list.append(sampled_language + " - " + sampled_level)

        if n_languages < max_:
            lang_list.extend(["-" for _ in range(n_languages, max_)])
        return lang_list

    def generate_certificates(self, all_certificates: list, min_: int = 0, max_: int = 3) -> str:
        max_cert = len(all_certificates)

        n_cert = random.choices(arange(min_, max_ + 1), k=1, weights=self.certificates_dist)[0]
        n_cert = min(n_cert, max_cert)

        r_idx_skill = choice(arange(0, max_cert), size=n_cert, replace=False)
        certificates = list(map(lambda x: all_certificates[x], r_idx_skill))

        if n_cert == 0:
            return "-"
        else:
            return ",".join(certificates)

    def get_curricula(self, size: int = 1, path: str = None) -> DataFrame:

        curricula = [self.__curriculum() for _ in tqdm(range(size))]

        curricula = pd.DataFrame(curricula).astype(
            dtype={"SmartWork": "int", "Experience_abroad": "int"})

        if path is not None:
            curricula.to_csv(path, index_label="kId")

        return curricula

    def get_offers(self, size: int = 1, path: str = None) -> DataFrame:

        offers = [self.__jobOffer() for _ in tqdm(range(size))]

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

        # randomly select one Job and retrieve the prototype
        job_name = random.choice(self.idx2jobName)
        current_job = self.jobs[job_name]

        # we impose the minimal instruction that the job requires
        degree = "P2" if current_job["degree"] else "P1"
        nationality = self.languages.sample(n=1, weights="P")["Languages"].values[0]

        skills = self.generate_skills(current_job["skills"])
        softskills = self.generate_skills(current_job["soft_skills"])

        education = self.education.sample(n=1, weights=degree)["Education"].values[0]
        # Min age based on a type of education
        min_age = self.education[self.education["Education"] == education]["Min_age"].values[0]
        min_age = max(min_age, current_job["age"][0])

        languages = self.generate_languages(nationality)

        cv = dict(
            IdealJob=job_name,  # 1
            Education=education,  # 2
            City=self.all_cities[random.randint(0, self.max_city - 1)],  # 3
            JobRange=choice(arange(30, 160, 10)),  # 4
            Skills0=skills[0],  # 5
            Skills1=skills[1],  # 6
            Skills2=skills[2],  # 7
            Skills3=skills[3],  # 8
            Skills4=skills[4],  # 9
            Softskills0=softskills[0],  # 10
            Softskills1=softskills[1],  # 11
            Softskills2=softskills[2],  # 12
            Softskills3=softskills[3],  # 13
            Softskills4=softskills[4],  # 14
            Age=random.randint(min_age, current_job["age"][1]),  # 15
            Language0=languages[0],  # 16
            Language1=languages[1],  # 17
            Language2=languages[2],  # 18
            Certificates=self.generate_certificates(current_job["certificates"])  # 19
        )

        yearExp = int(np.random.poisson(3))
        cv["YearExp"] = min(yearExp, cv["Age"] - min_age)  # 20

        # Retrieve min and max salary
        min_salary, max_salary = current_job["salary"]
        # The min salary for this kind of curriculum is given by
        # min_salary + 7% the salary of each year of experience
        min_salary += int((min_salary * 0.07) * cv["YearExp"])
        min_salary = min(min_salary, max_salary)

        cv["Ideal_Salary"] = int(random.randint(min_salary, max_salary) / 100) * 100  # 21
        cv["SmartWork"] = random.randint(0, 1) == 0 if current_job["smart_working"] else False  # 22
        cv["Experience_abroad"] = random.randint(0, 1) == 0  # 23

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

        # randomly select one Job and retrieve the prototype
        jobName = random.choice(self.idx2jobName)
        current_job = self.jobs[jobName]

        # we impose the minimal instruction that the job requires
        degree = "P2" if current_job["degree"] else "P1"
        nationality = self.languages.sample(n=1, weights="P")["Languages"].values[0]

        skills = self.generate_skills(current_job["skills"])
        softskills = self.generate_skills(current_job["soft_skills"])

        education = self.education.sample(n=1, weights=degree)["Education"].values[0]

        # Min age based on a type of education
        min_age = self.education[self.education["Education"] == education]["Min_age"].values[0]
        min_age = max(min_age, current_job["age"][0])
        max_age = random.randint(min_age, current_job["age"][1])

        languages = self.generate_languages(nationality)

        offer = dict(
            Job=jobName,  # 1
            Education=education,  # 2
            City=self.all_cities[random.randint(0, self.max_city - 1)],  # 3
            Skills0=skills[0],  # 4
            Skills1=skills[1],  # 5
            Skills2=skills[2],  # 6
            Skills3=skills[3],  # 7
            Skills4=skills[4],  # 8
            Softskills0=softskills[0],  # 9
            Softskills1=softskills[1],  # 10
            Softskills2=softskills[2],  # 11
            Softskills3=softskills[3],  # 12
            Softskills4=softskills[4],  # 13
            AgeMin=min_age,  # 14
            AgeMax=max_age,  # 15
            Language0=languages[0],  # 16
            Language1=languages[1],  # 17
            Language2=languages[2],  # 18
            Certificates=self.generate_certificates(current_job["certificates"])  # 19
        )

        yearExp = int(np.random.poisson(3))
        offer["YearExp"] = min(yearExp, max_age - min_age)  # 20

        # Retrieve min and max salary
        min_salary, max_salary = current_job["salary"]
        # The min salary for this kind of curriculum is given by
        # min_salary + 7% the salary of each year of experience
        min_salary += int((min_salary * 0.07) * offer["YearExp"])
        min_salary = min(min_salary, max_salary)

        offer["Offered_Salary"] = int(random.randint(min_salary, max_salary) / 100) * 100  # 21
        offer["SmartWork"] = random.randint(0, 1) == 0 if current_job["smart_working"] else False  # 22
        offer["Experience_abroad"] = random.randint(0, 1) == 0  # 23

        return offer
