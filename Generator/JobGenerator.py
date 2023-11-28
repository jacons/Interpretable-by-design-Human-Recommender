import copy
import csv
import os
import random
from typing import Literal

import numpy as np
import pandas as pd
from numpy import arange
from pandas import DataFrame, read_csv
from tqdm import tqdm

from Class_utils.JobGraph import JobGraph, RelationNode, TypeNode
from Class_utils.parameters import Language


class JobGenerator:
    def __init__(self, job_graph: JobGraph, sources: dict):
        """
        JobGenerator is a tool that allows us to generate synthetic data about the "curricula" and "job_offers"
        :param job_graph: Occupation-skill graph
        :param sources: dictionary that contains all paths used to load the resources

        opt_lang_distribution: Distribution of "number of optional languages for job-offer"

        cities_path: list of cities with their populations (used to sampling the city)

        languages_path: Source that represents all languages (used) with probability distribution
        languages_level_path : For each language in "language_path" it's defined a distribution of language level

        education_path: education levels distribution
        skill_synonyms_path: synonyms of skills (used to interchange the skill names)
        min_max_edu_occupation_path: Minimal education for isco group
        """
        # ------------------------ LOAD RESOURCES ------------------------
        self.kid_generator = self.__kid_generator()

        # --- Skills and Occupations ---
        self.job_graph = job_graph
        self.skills_synonyms = read_csv(sources["skill_synonyms_path"])

        # --- Cities ---
        self.all_cities = read_csv(sources["cities_path"], usecols=[0, 2]).astype({'city': 'string', 'P': 'float'})
        # --- Cities ---

        # --- Education  ---
        self.education = read_csv(sources["education_path"], index_col=0).astype(
            {'Education': 'string', 'Distribution': 'float', 'min_age': 'int'})
        min_max_edu = pd.read_csv(sources["min_max_edu_occupation_path"]).set_index("code")
        min_max_edu.index = min_max_edu.index.str[1:]
        self.min_max_edu = min_max_edu.to_dict("index")
        # --- Education  ---

        # --- Languages and levels  ---
        languages = read_csv(sources["languages_path"]).astype({"Languages": "string", "Prob": "float"})
        self.languages_level = read_csv(sources["languages_level_path"], index_col=0).astype(
            {"A1": "float", "A2": "float", "B1": "float", "B2": "float", "C1": "float", "C2": "float"})
        self.lang_prob = languages["Prob"].to_numpy()
        self.idx2language = languages["Languages"].to_dict()
        self.language2idx = {v: k for k, v in self.idx2language.items()}
        self.lang_level_dist = sources["opt_lang_distribution"]
        # --- Languages and levels  ---

        # ------------------------ LOAD RESOURCES ------------------------

    @staticmethod
    def __kid_generator():
        kid = 0
        while True:
            yield kid
            kid += 1

    def get_job_offers(self, size: int = 1) -> DataFrame:
        """
        It returns a list of syntetic job-offers
        :param size: Number of jobs-offers
        :param path:  destination path
        :param name: name of file
        """
        progress_bar = tqdm(range(size), desc="Generating the job-offers")
        offers = [self._jobOffer(idx) for idx in progress_bar]
        return pd.DataFrame(offers).set_index("qId")

    @staticmethod
    def save_job_offers(offers: DataFrame, path: str = None, name: str = None):
        if path is not None and name is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            offers.to_csv(f"{path}/{name}_job_offers.csv", quoting=csv.QUOTE_MINIMAL)

        return offers

    def _jobOffer(self, qId: int) -> dict:
        """
        Return a synthetic job-offer
        """
        # ------------------------------------------------------------------
        # randomly select one Job (id Occupation, Occupation name)
        id_occ, job_Name, isco_group = self.job_graph.sample_occupation()
        # ------------------------------------------------------------------
        edu_essential, edu_optional, min_age = self._generate_edu(isco_group=isco_group)
        # ------------------------------------------------------------------
        skills_es, skills_op, knowledge_es, knowledge_op = self._generate_skills(id_occ=id_occ)
        skills_es, skills_op = sorted(skills_es, reverse=True), sorted(skills_op, reverse=True)
        knowledge_es, knowledge_op = sorted(knowledge_es, reverse=True), sorted(knowledge_op, reverse=True)
        # ------------------------------------------------------------------
        min_age += random.randint(0, 5)
        max_age = min_age + random.randint(5, 20)
        # ------------------------------------------------------------------
        language_essential, language_optional = self._generate_languages()
        language_essential = sorted(language_essential, key=lambda x: x.name, reverse=True)
        # ------------------------------------------------------------------
        exp_essential = int(np.random.poisson(1.5))
        exp_essential = "-" if exp_essential == 0 else exp_essential
        exp_optional = True if random.random() <= 0.50 and exp_essential != "-" else False
        # ------------------------------------------------------------------
        offer = dict(
            qId=qId,  # 0
            Job=job_Name,  # 1
            metadata=dict(group=isco_group),  # 2

            Edu_essential=edu_essential,  # 3
            Edu_optional=edu_optional,  # 4
            AgeMin=min_age,  # 5
            AgeMax=max_age,  # 6
            City=self.all_cities.sample(n=1, weights="P")["city"].values[0],  # 7

            Competence_essential0=skills_es[0],  # 8
            Competence_essential1=skills_es[1],  # 9
            Competence_essential2=skills_es[2],  # 10
            Competence_essential3=skills_es[3],  # 11

            Competence_optional0=skills_op[0],  # 12
            Competence_optional1=skills_op[1],  # 13
            Competence_optional2=skills_op[2],  # 14

            Knowledge_essential0=knowledge_es[0],  # 15
            Knowledge_essential1=knowledge_es[1],  # 16
            Knowledge_essential2=knowledge_es[2],  # 17
            Knowledge_essential3=knowledge_es[3],  # 18

            Knowledge_optional0=knowledge_op[0],  # 19
            Knowledge_optional1=knowledge_op[1],  # 20
            Knowledge_optional2=knowledge_op[2],  # 21

            Language_essential0=language_essential[0].name,  # 22
            Language_essential1=language_essential[1].name,  # 23
            Language_optional0=language_optional[0].name,  # 24

            Language_level0=language_essential[0].level,  # 25
            Language_level1=language_essential[1].level,  # 26
            Language_level2=language_optional[0].level,  # 27
            Experience_essential=exp_essential,  # 28
            Experience_optional=exp_optional  # 29
        )
        return offer

    def _generate_edu(self, isco_group: str) -> tuple[str, str, int]:
        """
        Give an isco group return an "essential" and "optional" education and the minimal age
        """
        # retrieve minimal/maximal education for this kind of group
        min_edu = self.min_max_edu[isco_group[:3]]["min_edu"]  # minimal education
        max_edu = self.min_max_edu[isco_group[:3]]["max_edu"]  # maximal education

        # Sample an "essential education"
        education = self.education[(self.education.index >= min_edu) & (self.education.index <= max_edu)]
        id_educational = random.choices(education.index, weights=education["Distribution"])[0]
        edu_essential = self.education.loc[id_educational, "Education"]

        # Define an optional (Desirable) education
        edu_optional = "-"
        if random.random() >= 0.5 and id_educational <= 3:
            next_importance = id_educational + 1
            edu_optional = self.education.loc[next_importance, "Education"]

        min_age = self.education.loc[id_educational, "min_age"]
        return edu_essential, edu_optional, min_age

    def _generate_skills(self, id_occ: str, exclude_competence: list[str] = None,
                         exclude_knowledge: list[str] = None) -> tuple[list[str], list[str], list[str], list[str]]:
        """
        Given the id_occupation, it samples competences and knowledge
        and return : competence "essential", competence_op "optional", knowledge "essential", knowledge "optional"
        """
        if exclude_competence is None:
            exclude_competence = []
        if exclude_knowledge is None:
            exclude_knowledge = []

        # if exclude_competence and exclude_knowledge are empty
        min_ = 2 if len(exclude_competence) == 0 and len(exclude_knowledge) == 0 else 0

        competences_to_fill = 7 - len(exclude_competence)  # max 7 / min 3
        knowledge_to_fill = 7 - len(exclude_knowledge)  # max 7 / min 3

        essential_competences = random.randint(min_, 4)
        essential_knowledge = random.randint(min_, 4)

        max_opt_comp = competences_to_fill - essential_competences
        max_opt_know = knowledge_to_fill - essential_knowledge

        competence_es = self.job_graph.sample_skills(id_occ,
                                                     RelationNode.ES, TypeNode.SK,
                                                     exact_number=essential_competences, convert_ids=True,
                                                     exclude=exclude_competence)

        competence_op = self.job_graph.sample_skills(id_occ,
                                                     RelationNode.OP, TypeNode.SK,
                                                     min_=0, max_=max_opt_comp, convert_ids=True)

        knowledge_es = self.job_graph.sample_skills(id_occ,
                                                    RelationNode.ES, TypeNode.KN,
                                                    exact_number=essential_knowledge, convert_ids=True,
                                                    exclude=exclude_knowledge)

        knowledge_op = self.job_graph.sample_skills(id_occ,
                                                    RelationNode.OP, TypeNode.KN,
                                                    min_=0, max_=max_opt_know, convert_ids=True)

        return competence_es, competence_op, knowledge_es, knowledge_op

    def _generate_languages(self, e_lang=(1, 2), o_lang=(0, 1)) -> tuple[list[Language], list[Language]]:
        """
        Given the min/max essential language and min/max optional languages, it returns n (random) number
        of languages and their level.

        :param e_lang: (Min, max) essential language
        :param o_lang: (min, max) optional language

        """
        # Choose a number of essential languages (1 or 2)
        n_essential_lang = random.randint(e_lang[0], e_lang[1])
        # Choose a number of optional languages (0, 1) - with custom distribution
        n_optional_lang = random.choices(arange(o_lang[0], o_lang[1] + 1), weights=self.lang_level_dist)[0]

        # The Total number of languages -> min 1 max 3
        # mask is a boolean list with len equal to the number of languages to sample
        # it's containing a sequence of True/False. When True = it's an essential language, False = it's optional
        maks = [True] * n_essential_lang + [False] * n_optional_lang

        languages = []
        prob = self.lang_prob.copy()  # distribution of probability of all languages

        for ess in maks:  # pick n languages

            # convert "prob" into probability and choose an id_language
            id_language = random.choices(arange(0, len(prob)), weights=(prob / prob.sum()))[0]
            prob[id_language] = 0  # "0" probability mean that I cannot pick the same language again

            # Convert the id of the language into a name. E.g., 1 -> "Italian"
            name_language = self.idx2language[id_language]

            # with a certain probability, it's required at least a specific language level
            lang_level = "Any"
            if random.random() < 0.60 and ess:
                # Choose the level of language base on language
                lang_level = self.languages_level.T.sample().index.values[0]

            languages.append(Language(name_language, lang_level))

        language_essential = languages[0:n_essential_lang]
        language_optional = languages[n_essential_lang:]

        language_essential.extend([Language() for _ in range(n_essential_lang, e_lang[1])])
        language_optional.extend([Language() for _ in range(n_optional_lang, o_lang[1])])

        return language_essential, language_optional

    def generate_cvs(self, job_offers: DataFrame, mu: int = 100, std: int = 10):
        """
        Given a list of job-offers it produces a number of curricula for each job-offer
        :param job_offers: job-offers dataframe
        :param mu: average number of curricula per one job-offer
        :param std: standard deviation
        :param path: destination path
        :param name: name of file
        """
        curricula = []
        bar = tqdm(job_offers.itertuples(), total=len(job_offers), desc="Generating the curricula")
        for job_offer in bar:
            # select randomly the number of curricula to generate for this kind of occupation
            n_cvs = max(1, int(np.random.normal(mu, std)))

            # a certain percentage must be coherent with the job-offer requirement
            n_consistent_cv = int(n_cvs * 0.80)
            n_random_cv = n_cvs - n_consistent_cv

            curricula.extend(self._generate_consistent_cv(job_offer, n_consistent_cv))
            curricula.extend(self._generate_random_cv(job_offer[0], n_random_cv))

            bar.set_postfix(qId=job_offer[0])

        return DataFrame(curricula).set_index(keys="kId")

    @staticmethod
    def save_curricula(curricula: DataFrame, path: str = None, name: str = None) -> DataFrame:
        if path is not None and name is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            curricula.to_csv(f"{path}/{name}_curricula.csv", index=True, quoting=csv.QUOTE_MINIMAL)

        return curricula

    def _generate_consistent_cv(self, job_offer: tuple, n_consistent_cv: int) -> list[dict]:

        essential_competence = [job_offer[i] for i in range(8, 11 + 1) if job_offer[i] != "-"]
        essential_knowledge = [job_offer[i] for i in range(15, 18 + 1) if job_offer[i] != "-"]

        languages = [Language(job_offer[id_lang], job_offer[id_level])
                     for id_lang, id_level in zip([22, 23], [25, 26]) if job_offer[id_lang] != "-"]

        if len(essential_competence) == 0 or len(essential_knowledge) == 0:
            print(job_offer[0])

        curricula = []
        for _ in range(n_consistent_cv):
            # select a subsample of essential skills (competence and knowledge)
            competence = random.sample(essential_competence, k=random.randint(1, len(essential_competence)))
            knowledge = random.sample(essential_knowledge, k=random.randint(1, len(essential_knowledge)))

            # retrieve all jobs that have there skills
            similar_jobs = self.job_graph.get_job_with_skill(competence, knowledge)
            ideal_job = random.sample(similar_jobs, k=1)[0]
            curricula.append(
                self._get_curriculum(job_offer[0],  # qId
                                     ideal_job,
                                     job_offer[3],  # essential education
                                     job_offer[5], job_offer[6],  # min and max age
                                     competence,  # essential competences
                                     knowledge,  # essential knowledge
                                     copy.deepcopy(languages),  # essential language and level
                                     consistent=True)
            )
        return curricula

    def _generate_random_cv(self, qId: int, n_random_cv: int) -> list[dict]:
        return [self._get_curriculum(qId) for _ in range(n_random_cv)]

    def _generate_other_lang_from(self, languages: list[Language]) -> list[Language]:
        """
        Given a list of language and their level, it returns other a lst of other languages
        :param languages:
        :return:
        """

        # The essential languages are min 1 max 2.
        # With a certain probability p, we remove one language
        if len(languages) > 0 and random.random() >= 0.5:
            random.shuffle(languages)
            languages.pop()

        min_ = 1 if len(languages) == 0 else 0

        prob = self.lang_prob.copy()  # distribution of probability of all languages

        for lang in languages:
            #  Delete the probability to pick a language already sampled
            prob[self.language2idx[lang.name]] = 0

            # The candidate must satisfy the language. With a probability, the candidate
            # has a certification.
            if lang.level == "Any":
                if random.random() < 0.3:
                    lang.level = self.languages_level.T.sample(weights=lang.name).index.values[0]
                else:
                    lang.level = "Null"

        # the candidate has at max 3 language
        language_to_fill = 3 - len(languages)

        other_lang = random.randint(min_, language_to_fill)
        for _ in range(other_lang):
            id_language = random.choices(arange(0, len(prob)), weights=(prob / prob.sum()))[0]
            prob[id_language] = 0  # "0" probability mean that I cannot pick the same language again

            # Convert the id of the language into a name. E.g., 1 -> "Italian"
            name_lang = self.idx2language[id_language]

            # with a certain percentage p the candidate has a certificate
            lang_level = "Null"
            if random.random() < 0.3:
                # Choose the level of language base on language
                lang_level = self.languages_level.T.sample(weights=name_lang).index.values[0]
            languages.append(Language(name_lang, lang_level))

        languages.extend([Language() for _ in range(len(languages), 3)])

        return languages

    def _get_curriculum(self, qId: int, id_occ: str = None, edu_essential: str = "Less-than-degree",
                        min_age: int = 16, max_age: int = 60, competences: list[str] = None,
                        knowledge: list[str] = None, languages: list[Language] = None,
                        consistent: bool = False):

        # ------------------------------------------------------------------
        if competences is None:
            competences = []
        if knowledge is None:
            knowledge = []
        if languages is None:
            languages = []

        if id_occ is None:
            id_occ, _, group = self.job_graph.sample_occupation()
            min_edu = self.min_max_edu[group[:3]]["min_edu"]
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
            age = random.randint(edu_row["min_age"].values[0], 40)
        # ------------------------------------------------------------------

        new_comp_ess, new_comp_opt, new_know_ess, new_know_opt = self._generate_skills(id_occ, competences, knowledge)
        competences += new_comp_ess + new_comp_opt
        knowledge += new_know_ess + new_know_opt
        competences = sorted(competences, reverse=True)
        knowledge = sorted(knowledge, reverse=True)
        # ------------------------------------------------------------------
        languages = self._generate_other_lang_from(languages)
        languages = sorted(languages, key=lambda x: x.name, reverse=True)
        # ------------------------------------------------------------------
        experience = int(np.random.poisson(1.5))
        # ------------------------------------------------------------------
        cv = dict(
            qId=qId,  # 0
            kId=next(self.kid_generator),  # 0
            metadata=dict(occ=id_occ, consistent=consistent),  # 1
            Education=education,  # 2
            Age=age,  # 3
            City=self.all_cities.sample(n=1, weights="P")["city"].values[0],  # 4
            JobRange=int(np.random.randint(30, 100) / 10) * 10,  # 5

            Competences0=competences[0],  # 6
            Competences1=competences[1],  # 7
            Competences2=competences[2],  # 8
            Competences3=competences[3],  # 9
            Competences4=competences[4],  # 10
            Competences5=competences[5],  # 11
            Competences6=competences[6],  # 12

            Knowledge0=knowledge[0],  # 13
            Knowledge1=knowledge[1],  # 14
            Knowledge2=knowledge[2],  # 15
            Knowledge3=knowledge[3],  # 16
            Knowledge4=knowledge[4],  # 17
            Knowledge5=knowledge[5],  # 18
            Knowledge6=knowledge[6],  # 19

            Language0=languages[0].name,  # 20
            Language1=languages[1].name,  # 21
            Language2=languages[2].name,  # 22

            Language_level0=languages[0].level,  # 23
            Language_level1=languages[1].level,  # 24
            Language_level2=languages[2].level,  # 25
            Experience=experience  # 26
        )

        return cv

    def upgrade_with_synonymous(self, type_: Literal["cv", "offer"], df: DataFrame, p: float):
        """
        Given a dataframe, it's select a p% of the dataframe and substitute the competences/knowledge with synonyms
        :param type_:
        :param df:dataframe
        :param p: percentage of substitution
        """
        if df is None or p <= 0:
            return None

        if type_ == "cv":
            start_ = 6
            end_ = 19
        elif type_ == "offer":
            start_ = 7
            end_ = 20
        else:
            raise ValueError("Invalid 'type_' value. Use 'cv' or 'offer'.")

        progress_bar = tqdm(df.sample(frac=p).index, desc="Updating with synonyms...")
        for index in progress_bar:
            mask = [random.choice([True, False]) for _ in range(end_ - start_ + 1)]
            df.iloc[index, start_:end_ + 1] = self.job_graph.substitute_skills(mask, df.iloc[index, start_:end_ + 1])
