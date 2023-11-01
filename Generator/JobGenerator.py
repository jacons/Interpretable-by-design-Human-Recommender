import json
import random
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import arange
from numpy.random import choice
from pandas import DataFrame, read_csv
from tqdm import tqdm

from Generator.JobGraph import JobGraph


class JobGenerator:
    def __init__(self,
                 job2skills_path: str,
                 occupation_path: str,
                 skills_path: str,
                 cities_path: str,
                 languages_level_path: str,
                 languages_path: str,
                 education_path: str,
                 lang_level_distribution: Tuple[float],
                 certificates_distribution: Tuple[float]):
        """
        JobGenerator is a tool that allows us to generate synthetic data about the "curriculums" and "jobs offer"
        :param job2skills_path:
        :param occupation_path:
        :param skills_path:
        :param cities_path:
        :param languages_path:
        :param education_path:
        :param lang_level_distribution:
        :param certificates_distribution:
        """

        self.job_graph = JobGraph(job2skills_path, occupation_path, skills_path)

        self.certificates_dist = certificates_distribution
        self.lang_level_dist = lang_level_distribution

        self.all_cities = read_csv(cities_path, usecols=[0, 2]).astype(
            {'comune': 'string', 'P': 'float'})

        self.languages_level = read_csv(languages_level_path,index_col=0).astype(
            {"A1": "float", "A2": "float", "B1": "float", "B2": "float", "C1": "float", "C2": "float"})

        self.languages = read_csv(languages_path).astype(
            {"Languages": "string", "Prob": "int"})

        self.education = read_csv(education_path).astype(
            {'Importance': 'int', 'Education': 'string', 'P1': 'float',
             'P2': 'float', 'Min_age': 'int'})

        self.idx2language = {lang[0]: lang[1] for lang in self.languages.itertuples()}

    def jobOffer(self):
        # ------------------------------------------------------------------
        # randomly select one Job (id Occupation, Occupation name)
        id_occ, job_Name = self.job_graph.sample_occupation(convert_name=True)
        # ------------------------------------------------------------------
        # sample a "essential education"
        sample_educational = self.education.sample(n=1, weights="P2")
        edu_essential = sample_educational["Education"].values[0]
        importance = sample_educational["Importance"].values[0]

        # define a optional (Desirable) education
        edu_optional = "-"
        if random.randint(0, 1) == 0 and importance <= 3:
            edu_optional = self.education[self.education["Importance"] == importance + 1]["Education"].values[0]
        # ------------------------------------------------------------------
        skills_essential = self.job_graph.sample_skills(id_occ, "essential", convert_name=True)
        skills_optional = self.job_graph.sample_skills(id_occ, "essential", convert_name=True)
        # ------------------------------------------------------------------
        min_age = self.education[self.education["Education"] == edu_essential]["Min_age"].values[0]
        max_age = min_age + random.randint(5, 20)
        # ------------------------------------------------------------------
        prob = self.languages["Prob"].to_numpy()
        n_languages = random.choices(arange(1, 3 + 1), k=1, weights=self.lang_level_dist)[0]

        languages = []
        for n in n_languages:
            prob /= prob.sum()
            id_language = random.choices(arange(0, len(prob)), weights=prob)
            prob[id_language] = 0
            languages.append(id_language)

        languages = [self.idx2language[lang] for lang in languages]
        if n_languages < 3:
            languages.extend(["-" for _ in range(n_languages, 3)])

        language_levels = []
        for n in n_languages:
            language_levels.append(self.languages_level.T.sample(weights=languages[n]).index.values[0])

        if n_languages < 3:
            language_levels.extend(["-" for _ in range(n_languages, 3)])
        # ------------------------------------------------------------------
        offer = dict(
            Job=job_Name,  # 1
            Edu_essential=edu_essential,  # 2
            Edu_optional=edu_optional,  # 3
            City=self.all_cities.sample(n=1, weights="P")["comune"].values[0],  # 4
            Skills_essential0=skills_essential[0],  # 5
            Skills_essential1=skills_essential[1],  # 6
            Skills_essential2=skills_essential[2],  # 7
            Skills_essential3=skills_essential[3],  # 8
            Skills_essential4=skills_essential[4],  # 9
            Skills_essential5=skills_essential[5],  # 10
            Skills_optional0=skills_optional[0],  # 11
            Skills_optional1=skills_optional[1],  # 12
            Skills_optional2=skills_optional[2],  # 13
            Skills_optional3=skills_optional[3],  # 14
            Skills_optional4=skills_optional[4],  # 15
            Skills_optional5=skills_optional[5],  # 16
            AgeMin=min_age,  # 17
            AgeMax=max_age,  # 18
            Language0=languages[0],
            Language1=languages[1],
            Language2=languages[2],
            LanguageLevel0=language_levels[0],
            LanguageLevel1=language_levels[1],
            LanguageLevel2=language_levels[2],
        )

        """
        nationality = self.languages.sample(n=1, weights="P")["Languages"].values[0]

        skills = self.generate_skills(current_job["skills"])
        softskills = self.generate_skills(current_job["soft_skills"])



        # Min age based on a type of education
        min_age = self.education[self.education["Education"] == education]["Min_age"].values[0]
        min_age = max(min_age, current_job["age"][0])
        max_age = random.randint(min_age, current_job["age"][1])

        languages = self.generate_languages(nationality)

        offer = dict(
            Job=jobName,  # 1
            Education=education,  # 2
            City=self.all_cities.sample(n=1, weights="P")["comune"].values[0],  # 3
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

        yearExp = int(np.random.poisson(2))
        offer["YearExp"] = min(yearExp, max_age - min_age)  # 20

        offer["SmartWork"] = random.randint(0, 1) == 0 if current_job["smart_working"] else False  # 22
        offer["Experience_abroad"] = random.randint(0, 1) == 0  # 23

        return offer
        """


"""
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

    def get_offers(self, size: int = 1, path: str = None) -> DataFrame:

        offers = [self.__jobOffer() for _ in tqdm(range(size))]

        offers = pd.DataFrame(offers).astype(
            dtype={"SmartWork": "int", "Experience_abroad": "int"})

        if path is not None:
            offers.to_csv(path, index_label="qId")

        return offers

    def get_curricula(self, size: int = 1, path: str = None) -> DataFrame:

        curricula = [self.__curriculum() for _ in tqdm(range(size))]

        curricula = pd.DataFrame(curricula).astype(
            dtype={"SmartWork": "int", "Experience_abroad": "int"})

        if path is not None:
            curricula.to_csv(path, index_label="kId")

        return curricula

    def __curriculum(self) -> dict:

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
            City=self.all_cities.sample(n=1, weights="P")["comune"].values[0],  # 3
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

        yearExp = int(np.random.poisson(2))
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
"""
