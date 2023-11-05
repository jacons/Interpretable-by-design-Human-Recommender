import random

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
    def __init__(self, sources: dict):
        # JobGenerator is a tool that allows us to generate synthetic data about the "curricula" and "job_offers"

        """
        job2skills_path: relation "many-to-many" between occupation and skills/knowledge
        occupation_path: dt of occupations and their info
        skills_path: dt of skills and their info
        cities_path: list of cities with their populations
        languages_path: italian languages distribution
        lang_level_distribution: optional languages distribution
        min_edu_occupation_path: Minimal education for isco group
        education_path: education level distribution
        """

        self.job_graph = JobGraph(sources["job2skills_path"],
                                  sources["occupation_path"],
                                  sources["skills_path"])

        self.lang_level_dist = sources["lang_level_distribution"]
        self.kid_generator = kid_generator()

        self.all_cities = read_csv(sources["cities_path"], usecols=[0, 2]).astype({'comune': 'string', 'P': 'float'})
        self.languages = read_csv(sources["languages_path"]).astype({"Languages": "string", "Prob": "float"})

        self.languages_level = read_csv(sources["languages_level_path"], index_col=0).astype(
            {"A1": "float", "A2": "float", "B1": "float", "B2": "float", "C1": "float", "C2": "float"})

        self.education = read_csv(sources["education_path"], index_col=0).astype(
            {'Education': 'string', 'Distribution': 'float', 'Min_age': 'int'})

        min_ed = read_csv(sources["min_edu_occupation_path"]).astype(
            {"id_group": "int", "id_edu": "int"}
        )

        # dictionary isco_group -> minimal edu
        self.min_edu = {i[1]: i[2] for i in min_ed.itertuples()}
        self.idx2language = {lang[0]: lang[1] for lang in self.languages.itertuples()}
        self.language2idx = {lang[1]: lang[0] for lang in self.languages.itertuples()}

    def generate_edu(self, isco_group: int) -> tuple[str, str, int]:
        """
        Give an isco group return an "essential" and "optional" education; and the minimal age
        """
        # retrieve minimal_education for this kind of group
        min_edu = self.min_edu[isco_group // 1000]

        # Sample an "essential education"
        education = self.education[self.education.index >= min_edu]
        id_educational = random.choices(education.index, weights=education["Distribution"])[0]
        edu_essential = self.education.loc[id_educational, "Education"]

        # Define an optional (Desirable) education
        edu_optional = "-"
        if random.random() >= 0.5 and id_educational <= 3:
            next_importance = id_educational + 1
            edu_optional = self.education.loc[next_importance, "Education"]

        min_age = self.education.loc[id_educational, "Min_age"]
        return edu_essential, edu_optional, min_age

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
        n_optional_lang = random.choices(arange(o_lang[0], o_lang[1] + 1), weights=self.lang_level_dist)[0]

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
        id_occ, job_Name, isco_group = self.job_graph.sample_occupation()
        # ------------------------------------------------------------------
        edu_essential, edu_optional, min_age = self.generate_edu(isco_group=isco_group)
        # ------------------------------------------------------------------
        skills_es, skills_op, knowledge_es, knowledge_op = self.generate_skills(id_occ)
        # ------------------------------------------------------------------
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
            Knoleadge_essential3=knowledge_es[3],  # 17

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
        offers = pd.DataFrame(offers).set_index("qId")

        if path is not None:
            offers.to_csv(path)

        return offers

    def generate_cvs(self, job_offers: DataFrame, mu: int = 100, sig: int = 10):
        curricula = []
        bar = tqdm(job_offers.itertuples(), total=len(job_offers))
        for job_offer in bar:

            # select randomly the number of curricula to generate for this kind of occupation
            n_cvs = int(np.random.normal(mu, sig))
            # a certain percentage must be coherent with the job-offer requirement
            n_consistent_job = int(n_cvs * 0.80)

            essential_competence = [job_offer[i] for i in range(7, 10 + 1) if job_offer[i] != "-"]
            essential_knowledge = [job_offer[i] for i in range(14, 17 + 1) if job_offer[i] != "-"]
            essential_language = [job_offer[i] for i in [21, 22] if job_offer[i] != "-"]
            essential_lang_level = [job_offer[i] for i in [24, 25] if job_offer[i] != "-"]

            for _ in range(n_consistent_job):
                # select a subsample of essential skills (competence and knowledge)

                competence, knowledge = [], []
                if len(essential_competence) > 0:
                    competence = random.sample(essential_competence,
                                               k=random.randint(1, len(essential_competence)))
                if len(essential_knowledge) > 0:
                    knowledge = random.sample(essential_knowledge,
                                              k=random.randint(1, len(essential_knowledge)))

                # retrieve all jobs that have there skills
                similar_jobs = self.job_graph.get_job_with_skill(competence, knowledge)
                ideal_job = random.sample(similar_jobs)

                curricula.append(
                    self.get_curriculum(job_offer[0],  # qId
                                        ideal_job,
                                        job_offer[2],  # essential education
                                        job_offer[4], job_offer[5],  # min and max age
                                        competence,  # essential competences
                                        knowledge,  # essential knowledge
                                        essential_language.copy(),  # essential language
                                        essential_lang_level.copy(),  # essential language level
                                        job_offer[27])  # essential experience
                )
            for _ in range(n_cvs - n_consistent_job):
                curricula.append(self.get_curriculum(job_offer[0]))
            bar.set_postfix(qId=job_offer[0])
        return DataFrame(curricula)

    def generate_other_skill_from(self, id_occ: str, competences: list[str], knowledge: list[str]):

        competences_to_fill = 7 - len(competences)  # max 5 / min 3
        knowledge_to_fill = 7 - len(knowledge)  # max 5 / min 3

        other_essential_competences = random.randint(0, competences_to_fill)
        other_essential_knowledge = random.randint(0, knowledge_to_fill)

        optional_competences = competences_to_fill - other_essential_competences
        optional_knowledge = knowledge_to_fill - other_essential_knowledge

        new_competences, new_knowledge = [], []
        new_competences.extend(self.job_graph.sample_skills(id_occ,
                                                            relation=RelationNode.ES, type_node=TypeNode.SK,
                                                            min_=0, max_=other_essential_competences,
                                                            convert_ids=True, exclude=competences))
        new_competences.extend(self.job_graph.sample_skills(id_occ,
                                                            relation=RelationNode.OP, type_node=TypeNode.SK,
                                                            min_=0, max_=optional_competences,
                                                            convert_ids=True))
        new_knowledge.extend(self.job_graph.sample_skills(id_occ,
                                                          relation=RelationNode.ES, type_node=TypeNode.KN,
                                                          min_=0, max_=other_essential_knowledge,
                                                          convert_ids=True, exclude=knowledge))
        new_knowledge.extend(self.job_graph.sample_skills(id_occ,
                                                          relation=RelationNode.OP, type_node=TypeNode.KN,
                                                          min_=0, max_=optional_knowledge,
                                                          convert_ids=True))
        return new_competences, new_knowledge

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
                       edu_essential: str = "Less-than-degree",
                       min_age: int = 16, max_age: int = 60,
                       competences: list[str] = None,  # essential skills
                       knowledge: list[str] = None,  # essential skills
                       languages: list[str] = None,  # essential languages
                       langs_level: list[str] = None,  # essential language's level
                       experience: int | str = "-"):

        # ------------------------------------------------------------------
        if competences is None:
            competences = []
        if knowledge is None:
            knowledge = []
        if langs_level is None:
            langs_level = []
        if languages is None:
            languages = []

        if id_occ is None:
            id_occ, _, group = self.job_graph.sample_occupation()
            min_edu = self.min_edu[group // 1000]
            edu_essential = self.education.loc[min_edu, "Education"]

        edu_row = self.education[self.education["Education"] == edu_essential]
        min_edu = edu_row.index.values[0]
        educations = self.education[self.education.index >= min_edu]
        id_educational = random.choices(educations.index, weights=educations["Distribution"])[0]
        education = self.education.loc[id_educational, "Education"]
        # ------------------------------------------------------------------
        if random.random() <= 0.80:
            age = random.randint(min_age, max_age)
        else:
            age = random.randint(edu_row["Min_age"].values[0], 60)
        # ------------------------------------------------------------------
        new_competences, new_knowledge = self.generate_other_skill_from(id_occ, competences, knowledge)
        new_competences += competences
        new_knowledge += knowledge
        # ------------------------------------------------------------------
        languages, langs_level = self.generate_other_lang_from(languages, langs_level)
        # ------------------------------------------------------------------
        experience = int(np.random.poisson(1.5)) if experience == "-" else int(experience) + int(np.random.poisson(1.5))
        # ------------------------------------------------------------------
        cv = dict(
            qId=qId,  # 0
            kId=next(self.kid_generator),  # 1
            Education=education,  # 2
            Age=age,  # 3
            City=self.all_cities.sample(n=1, weights="P")["comune"].values[0],  # 4
            JobRange=int(np.random.randint(30, 100) / 10) * 10,  # 5

            Competences0=new_competences[0],  # 6
            Competences1=new_competences[1],  # 7
            Competences2=new_competences[2],  # 8
            Competences3=new_competences[3],  # 9
            Competences4=new_competences[4],  # 10
            Competences5=new_competences[5],  # 11
            Competences6=new_competences[6],  # 12

            Knowledge0=new_knowledge[0],  # 13
            Knowledge1=new_knowledge[1],  # 14
            Knowledge2=new_knowledge[2],  # 15
            Knowledge3=new_knowledge[3],  # 16
            Knowledge4=new_knowledge[4],  # 17
            Knowledge5=new_knowledge[5],  # 18
            Knowledge6=new_knowledge[6],  # 19

            Language0=languages[0],  # 20
            Language1=languages[1],  # 21
            Language2=languages[2],  # 22

            Language_level0=langs_level[0],  # 23
            Language_level1=langs_level[1],  # 24
            Language_level2=langs_level[2],  # 25
            Experience=experience  # 26
        )

        return cv
