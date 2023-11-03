import random
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import arange
from pandas import DataFrame, read_csv
from tqdm import tqdm

from Generator.JobGraph import JobGraph, RelationNode, TypeNode


def kid_generator():
    kid = 0
    while True:
        yield kid
        kid += 1


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
        JobGenerator is a tool that allows us to generate synthetic data about the "curriculums" and "job_offers"
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
        self.kid_generator = kid_generator()
        self._load_data(cities_path, languages_level_path, languages_path, education_path)

    def _load_data(self, cities_path, languages_level_path, languages_path, education_path):
        self.all_cities = read_csv(cities_path, usecols=[0, 2]).astype({'comune': 'string', 'P': 'float'})

        self.languages_level = read_csv(languages_level_path, index_col=0).astype(
            {"A1": "float", "A2": "float", "B1": "float", "B2": "float", "C1": "float", "C2": "float"})

        self.languages = read_csv(languages_path).astype({"Languages": "string", "Prob": "int"})

        self.education = read_csv(education_path, index_col=0).astype(
            {'Education': 'string', 'P1': 'float', 'P2': 'float', 'Min_age': 'int'})

        self.idx2language = {lang[0]: lang[1] for lang in self.languages.itertuples()}
        self.language2idx = {lang[1]: lang[0] for lang in self.languages.itertuples()}

    def generate_edu(self):
        # Sample an "essential education"
        sample_educational = random.choices(self.education.index, k=1, weights=self.education["P2"])[0]
        edu_essential = self.education.loc[sample_educational, "Education"]
        importance = sample_educational

        # Define an optional (Desirable) education
        edu_optional = "-"
        if random.random() >= 0.5 and importance <= 3:
            next_importance = importance + 1
            edu_optional = self.education.loc[next_importance, "Education"]

        return edu_essential, edu_optional

    def generate_skills(self, id_occ: str):
        skills_es = self.job_graph.sample_skills(id_occ,
                                                 RelationNode.ES, TypeNode.SK,
                                                 min_=2, max_=4,
                                                 convert_ids=True)

        skills_op = self.job_graph.sample_skills(id_occ,
                                                 RelationNode.OP, TypeNode.SK,
                                                 min_=0, max_=3,
                                                 convert_ids=True)

        knowledge_es = self.job_graph.sample_skills(id_occ,
                                                    RelationNode.ES, TypeNode.KN,
                                                    min_=2, max_=4,
                                                    convert_ids=True)

        knowledge_op = self.job_graph.sample_skills(id_occ,
                                                    RelationNode.OP, TypeNode.KN,
                                                    min_=0, max_=3,
                                                    convert_ids=True)

        return skills_es, skills_op, knowledge_es, knowledge_op

    def generate_languages(self, e_lang=(1, 2), o_lang=(0, 1)):
        # chose a number of essential languages (1 or 2)
        n_essential_lang = random.randint(e_lang[0], e_lang[1])
        # chose a number of optional languages (0, 1)
        n_optional_lang = random.choices(arange(o_lang[0], o_lang[1] + 1), k=1, weights=self.lang_level_dist)[0]

        # total number of languages min 1 max 3
        n_languages = n_optional_lang + n_essential_lang

        languages, language_levels = [], []
        prob = self.languages["Prob"].to_numpy().copy()  # distribution of probability of all languages

        for n in range(n_languages):  # pick n languages
            # convert "prob" into probability and chose an id_language
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

    def __jobOffer(self, qId: int) -> dict:
        # ------------------------------------------------------------------
        # randomly select one Job (id Occupation, Occupation name)
        id_occ, job_Name = self.job_graph.sample_occupation()
        # ------------------------------------------------------------------
        edu_essential, edu_optional = self.generate_edu()
        # ------------------------------------------------------------------
        skills_es, skills_op, knowledge_es, knowledge_op = self.generate_skills(id_occ)
        # ------------------------------------------------------------------
        min_age = self.education[self.education["Education"] == edu_essential]["Min_age"].values[0]
        max_age = min_age + random.randint(5, 20)
        # ------------------------------------------------------------------
        language_essential, language_optional, language_levels = self.generate_languages()
        # ------------------------------------------------------------------
        exp_essential = int(np.random.poisson(1.5))
        exp_essential = "-" if exp_essential == 0 else exp_essential
        exp_optional = True if random.random() <= 0.50 else False
        # ------------------------------------------------------------------
        offer = dict(
            qId=qId,  # 0
            Job=job_Name,  # 1
            Edu_essential=edu_essential,  # 2
            Edu_optional=edu_optional,  # 3
            AgeMin=min_age,  # 4
            AgeMax=max_age,  # 5
            City=self.all_cities.sample(n=1, weights="P")["comune"].values[0],  # 6

            Competence_essential0=skills_es[0],  # 7
            Competence_essential1=skills_es[1],  # 8
            Competence_essential2=skills_es[2],  # 9
            Competence_essential3=skills_es[3],  # 10
            Competence_optional0=skills_op[0],  # 11
            Competence_optional1=skills_op[1],  # 12
            Competence_optional2=skills_op[2],  # 13
            Knoleadge_essential0=knowledge_es[0],  # 14
            Knoleadge_essential1=knowledge_es[1],  # 15
            Knoleadge_essential2=knowledge_es[2],  # 16
            Knoleadge_essential3=knowledge_es[3],  # 16
            Knoleadge_optional0=knowledge_op[0],  # 18
            Knoleadge_optional1=knowledge_op[1],  # 19
            Knoleadge_optional2=knowledge_op[2],  # 20
            Language_essential0=language_essential[0],  # 21
            Language_essential1=language_essential[1],  # 22
            Language_optional0=language_optional[0],  # 23
            Language_level0=language_levels[0],  # 24
            Language_level1=language_levels[1],  # 25
            Language_level2=language_levels[2],  # 26
            Experience_essential=exp_essential,  # 27
            Experience_optional=exp_optional  # 28
        )
        return offer

    def get_offers(self, size: int = 1, path: str = None) -> DataFrame:

        offers = [self.__jobOffer(idx) for idx in tqdm(range(size))]
        offers = pd.DataFrame(offers)

        if path is not None:
            offers.to_csv(path, index=False)

        return offers

    def generate_cvs(self, job_offers: DataFrame):
        consistent_cv = []
        for job_offer in job_offers.itertuples():

            n_cvs = int(np.random.normal(100, 10))
            n_consistent_job = int(n_cvs * 0.80)

            essential_skill = [job_offer[i] for i in range(7, 12 + 1) if job_offer[i] != "-"]
            essential_language = [job_offer[i] for i in range(19, 20 + 1) if job_offer[i] != "-"]
            essential_lang_level = [job_offer[i] for i in range(22, 23 + 1) if job_offer[i] != "-"]

            similar_jobs = self.job_graph.get_job_with_skill(essential_skill)

            for _ in range(n_consistent_job):
                ideal_job = random.sample(similar_jobs, 1)[0]
                consistent_cv.append(
                    self.get_curriculum(job_offer[0],  # kId
                                        ideal_job,
                                        job_offer[2],  # essential education
                                        job_offer[4], job_offer[5],  # min and max age
                                        essential_skill.copy(),
                                        essential_language.copy(),  # essential language
                                        essential_lang_level.copy(),  # essential language level
                                        job_offer[25])  # essential experience
                )
            # for _ in range(n_cvs - n_consistent_job):
            #     consistent_cv.append(
            #         self.get_curriculum(job_offer[0])
            #     )
        return DataFrame(consistent_cv)

    def generate_other_skill_from(self, id_occ: str, skills: list):

        skill_to_fill = 8 - len(skills)  # max 6 / min 2

        other_essential_skills = random.randint(0, skill_to_fill)
        optional_skills = skill_to_fill - other_essential_skills

        skills.extend(
            self.job_graph.sample_skills(id_occ,
                                         relation_type="essential",
                                         min_=0, max_=other_essential_skills,
                                         convert_ids=True, exclude=skills)
        )
        skills.extend(
            self.job_graph.sample_skills(id_occ,
                                         relation_type="optional",
                                         min_=0, max_=optional_skills,
                                         convert_ids=True)
        )
        return skills

    def generate_other_lang_from(self, languages: list, langs_level: list):
        language_to_fill = 3 - len(languages)

        prob = self.languages["Prob"].to_numpy().copy()  # distribution of probability of all languages

        for lang in languages:
            prob[self.language2idx[lang]] = 0

        other_lang = random.randint(0, language_to_fill)
        for _ in range(other_lang):
            id_language = random.choices(arange(0, len(prob)), weights=(prob / prob.sum()))[0]
            prob[id_language] = 0  # "0" probability mean that I cannot pick the same language again
            name_lang = self.idx2language[id_language]
            languages.append(name_lang)

            # Chose the level of language base on language
            lang_level = self.languages_level.T.sample(weights=name_lang).index.values[0]
            langs_level.append(lang_level)

        diff = len(languages)
        if diff < 3:
            languages.extend(["-" for _ in range(diff, 3)])
            langs_level.extend(["-" for _ in range(diff, 3)])

        return languages, langs_level

    def get_curriculum(self,
                       qId: int,
                       id_occ: str = None,
                       edu_essential: str = None,
                       min_age: int = None, max_age: int = None,
                       skills: list = None,  # essential skills
                       languages: list = None,  # essential languages
                       langs_level: list = None,  # essential language's level
                       experience: int = None):

        # ------------------------------------------------------------------
        edu_row = self.education[self.education["Education"] == edu_essential]
        importance = edu_row["Importance"].values[0]
        education = self.education[self.education["Importance"] >= importance].sample()["Education"].values[0]
        # ------------------------------------------------------------------
        if random.random() <= 0.80:
            age = random.randint(min_age, max_age)
        else:
            age = random.randint(edu_row["Min_age"].values[0], 60)
        # ------------------------------------------------------------------
        skills = self.generate_other_skill_from(id_occ, skills)
        # ------------------------------------------------------------------
        languages, langs_level = self.generate_other_lang_from(languages, langs_level)
        # ------------------------------------------------------------------
        experience = int(np.random.poisson(1.5)) if experience == "-" else experience + int(np.random.poisson(1.5))
        # ------------------------------------------------------------------
        cv = dict(
            qId=qId,
            kId=next(self.kid_generator),
            Education=education,  # 1
            Age=age,  # 2
            City=self.all_cities.sample(n=1, weights="P")["comune"].values[0],  # 3
            JobRange=np.random.randint(30, 100),
            Skills0=skills[0],  # 4
            Skills1=skills[1],  # 5
            Skills2=skills[2],  # 6
            Skills3=skills[3],  # 7
            Skills4=skills[4],  # 8
            Skills5=skills[5],  # 9
            Skills6=skills[6],  # 10
            Skills7=skills[7],  # 11
            Language0=languages[0],  # 12
            Language1=languages[1],  # 13
            Language2=languages[2],  # 14
            Language_level0=langs_level[0],  # 15
            Language_level1=langs_level[1],  # 16
            Language_level2=langs_level[2],  # 17
            Experience=experience,  # 18
        )

        return cv
