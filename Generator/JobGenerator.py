import random
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import arange
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
                 lang_level_distribution: Tuple[float]):
        """
        JobGenerator is a tool that allows us to generate synthetic data about the "curriculums" and "job offers"
        :param job2skills_path:
        :param occupation_path:
        :param skills_path:
        :param cities_path:
        :param languages_path:
        :param education_path:
        :param lang_level_distribution:
        """

        self.job_graph = JobGraph(job2skills_path, occupation_path, skills_path)

        self.lang_level_dist = lang_level_distribution

        self.all_cities = read_csv(cities_path, usecols=[0, 2]).astype(
            {'comune': 'string', 'P': 'float'})

        self.languages_level = read_csv(languages_level_path, index_col=0).astype(
            {"A1": "float", "A2": "float", "B1": "float", "B2": "float", "C1": "float", "C2": "float"})

        self.languages = read_csv(languages_path).astype(
            {"Languages": "string", "Prob": "int"})

        self.education = read_csv(education_path).astype(
            {'Importance': 'int', 'Education': 'string', 'P1': 'float',
             'P2': 'float', 'Min_age': 'int'})

        self.idx2language = {lang[0]: lang[1] for lang in self.languages.itertuples()}

    def generate_edu(self):
        # sample a "essential education"
        sample_educational = self.education.sample(n=1, weights="P2")
        edu_essential = sample_educational["Education"].values[0]
        importance = sample_educational["Importance"].values[0]

        # define a optional (Desirable) education
        edu_optional = "-"
        if random.randint(0, 1) == 0 and importance <= 3:
            edu_optional = self.education[self.education["Importance"] == importance + 1]["Education"].values[0]
        return edu_essential, edu_optional

    def generate_skill(self, id_occ: str, min_: int = 2, max_: int = 6):
        skills_essential = self.job_graph.sample_skills(id_occ, "essential",
                                                        min_=min_,
                                                        max_=max_,
                                                        convert_name=True)

        skills_optional = self.job_graph.sample_skills(id_occ, "optional",
                                                       min_=min_,
                                                       max_=max_,
                                                       convert_name=True)
        return skills_essential, skills_optional

    def generate_languages(self, e_lang=(1, 2), o_lang=(0, 2)):
        # chose a number of essential languages (1 or 2)
        n_essential_lang = random.randint(e_lang[0], e_lang[1])
        # chose a number of optional languages (0, 1, 2)
        n_optional_lang = random.choices(arange(o_lang[0], o_lang[1] + 1), k=1, weights=self.lang_level_dist)[0]

        # total number of languages min 1 max 4
        n_languages = n_optional_lang + n_essential_lang

        languages, language_levels = [], []
        prob = self.languages["Prob"].to_numpy().copy()  # distribution of probability of all languages

        for n in range(n_languages):  # pick n languages
            # convert "prob" into probability
            id_language = random.choices(arange(0, len(prob)), weights=(prob / prob.sum()))[0]
            prob[id_language] = 0  # "0" probability mean that I cannot pick the same language again
            languages.append(self.idx2language[id_language])

            # Chose the level of language base on language
            lang_level = self.languages_level.T.sample(weights=languages[n]).index.values[0]
            language_levels.append(lang_level)

        language_essential = languages[0:n_essential_lang]
        language_optional = languages[n_essential_lang:]

        language_essential.extend(["-" for _ in range(n_essential_lang, e_lang[1])])
        language_optional.extend(["-" for _ in range(n_optional_lang, e_lang[1])])
        language_levels.extend(["-" for _ in range(n_languages, e_lang[1] + o_lang[1])])

        return language_essential, language_optional, language_levels

    def __jobOffer(self) -> dict:
        # ------------------------------------------------------------------
        # randomly select one Job (id Occupation, Occupation name)
        id_occ, job_Name = self.job_graph.sample_occupation(convert_name=True)
        # ------------------------------------------------------------------
        edu_essential, edu_optional = self.generate_edu()
        # ------------------------------------------------------------------
        skills_essential, skills_optional = self.generate_skill(id_occ)
        # ------------------------------------------------------------------
        min_age = self.education[self.education["Education"] == edu_essential]["Min_age"].values[0]
        max_age = min_age + random.randint(5, 20)
        # ------------------------------------------------------------------
        language_essential, language_optional, language_levels = self.generate_languages()
        # ------------------------------------------------------------------
        exp_essential = int(np.random.poisson(1.5))
        exp_essential = "-" if exp_essential == 0 else exp_essential
        exp_optional = exp_essential if random.randint(0, 1) == 1 else "-"
        # ------------------------------------------------------------------
        offer = dict(
            Job=job_Name,  # 1
            Edu_essential=edu_essential,  # 2
            Edu_optional=edu_optional,  # 3
            AgeMin=min_age,  # 4
            AgeMax=max_age,  # 5
            City=self.all_cities.sample(n=1, weights="P")["comune"].values[0],  # 6
            Skills_essential0=skills_essential[0],  # 7
            Skills_essential1=skills_essential[1],  # 8
            Skills_essential2=skills_essential[2],  # 9
            Skills_essential3=skills_essential[3],  # 10
            Skills_essential4=skills_essential[4],  # 11
            Skills_essential5=skills_essential[5],  # 12
            Skills_optional0=skills_optional[0],  # 13
            Skills_optional1=skills_optional[1],  # 14
            Skills_optional2=skills_optional[2],  # 15
            Skills_optional3=skills_optional[3],  # 16
            Skills_optional4=skills_optional[4],  # 17
            Skills_optional5=skills_optional[5],  # 18
            Language_essential0=language_essential[0],  # 19
            Language_essential1=language_essential[1],  # 20
            Language_optional0=language_optional[0],  # 21
            Language_optional1=language_optional[1],  # 22
            Language_level0=language_levels[0],  # 23
            Language_level1=language_levels[1],  # 24
            Language_level2=language_levels[2],  # 25
            Language_level3=language_levels[3],  # 26
            Experience_essential=exp_essential,  # 27
            Experience_optional=exp_optional  # 28
        )
        return offer

    def get_offers(self, size: int = 1, path: str = None) -> DataFrame:

        offers = [self.__jobOffer() for _ in tqdm(range(size))]

        offers = pd.DataFrame(offers)

        if path is not None:
            offers.to_csv(path, index_label="qId")

        return offers


"""


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
