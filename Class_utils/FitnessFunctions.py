from typing import Iterable

from pandas import DataFrame
from tqdm import tqdm

from Class_utils.FitnessClasses import *
from Class_utils.parameters import Language
from KnowledgeBase.JobGraph import JobGraph


class FitnessFunctions:
    def __init__(self, sources: dict, job_graph: JobGraph = None):

        self.fitness_cities = FitnessCity(sources["cities_dist"])
        self.fitness_age = FitnessAge()
        self.fitness_experience = FitnessExperience()
        self.fitness_edu = FitnessEdu(sources["education_path"])
        self.fitness_languages = FitnessLanguages()
        self.fitness_skills = FitnessSkills(job_graph)

    @staticmethod
    def filter(list_: Iterable[str]) -> list[str]:
        return [item for item in list_ if item != "-"]

    def generate_fitness_score(self, offers: DataFrame, curricula: DataFrame) -> DataFrame:
        dataset = []
        bar = tqdm(offers.itertuples(), total=len(offers), desc="Generating the fitness scores")
        for offer in bar:
            curricula_ = curricula[curricula["qId"] == offer[0]]
            for cv in curricula_.itertuples():
                dataset.append(self.fitness(offer, cv))
            bar.set_postfix(qId=offer[0])

        return DataFrame(data=dataset).astype({"qId": "int", "kId": "int"})

    def fitness(self, offer: tuple, cv: tuple) -> dict:

        cv_lang = [Language(*lang) for lang in cv[9]]
        of_lang_ess = [Language(*lang) for lang in offer[12]]
        of_lang_opt = [Language(*lang) for lang in offer[13]]
        of_lang_opt = Language() if len(of_lang_opt) == 0 else of_lang_opt[0]

        of_comp_ess, of_comp_opt = offer[8], offer[9]
        of_know_ess, of_know_opt = offer[10], offer[11]
        cv_comp, cv_know = cv[7], cv[8]

        fit_edu_basic = self.fitness_edu.fitness_basic(offer[3], cv[3])
        fit_edu_bonus = self.fitness_edu.fitness_bonus(offer[4], cv[3])
        fit_exp_basic = self.fitness_experience.fitness_basic(offer[14], cv[10])
        fit_exp_bonus = self.fitness_experience.fitness_bonus(offer[14], offer[15], cv[10])
        fit_lang_basic = self.fitness_languages.fitness_basic(of_lang_ess, cv_lang)
        fit_lang_bonus = self.fitness_languages.fitness_bonus(cv_lang, of_lang_opt)
        fit_competence = self.fitness_skills.fitness(of_comp_ess, of_comp_opt, cv_comp)
        fit_knowledge = self.fitness_skills.fitness(of_know_ess, of_know_opt, cv_know)

        result = dict(
            qId=offer[0],
            kId=cv[0],
            info="",
            fitness_edu_basic=fit_edu_basic,
            fitness_edu_bonus=fit_edu_bonus,
            fitness_city=self.fitness_cities.fitness(offer[7], cv[5], cv[6]),
            fitness_age=self.fitness_age.fitness_basic(cv[4], offer[5], offer[6]),
            fitness_exp_basic=fit_exp_basic,
            fitness_exp_bonus=fit_exp_bonus,
            fitness_lang_basic=fit_lang_basic["score_language"],
            fitness_lang_lvl_basic=fit_lang_basic["score_level"],
            fitness_lang_bonus=fit_lang_bonus,
            fitness_comp_essential=fit_competence["score_essential"],
            fitness_comp_sim_essential=fit_competence["score_similarity_essential"],
            fitness_comp_optional=fit_competence["score_optional"],
            fitness_comp_sim_bonus=fit_competence["score_similarity_optional"],
            fitness_know_essential=fit_knowledge["score_essential"],
            fitness_know_sim_essential=fit_knowledge["score_similarity_essential"],
            fitness_know_optional=fit_knowledge["score_optional"],
            fitness_know_sim_optional=fit_knowledge["score_similarity_optional"],
        )
        return result
