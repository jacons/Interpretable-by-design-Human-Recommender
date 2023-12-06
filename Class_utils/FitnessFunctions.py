from typing import Iterable

from pandas import DataFrame
from tqdm import tqdm

from Class_utils.FitnessClasses import *
from Class_utils.parameters import Language


class FitnessFunctions:
    def __init__(self, sources: dict, job_graph: JobGraph = None):

        self.fitness_cities = FitnessCity(sources["cities_dist"])
        self.fitness_age = FitnessAge()
        self.fitness_experience = FitnessExperience()
        self.fitness_edu = FitnessEdu(sources["education_path"])
        self.fitness_languages = FitnessLanguages()
        self.fitness_skills = FitnessSkills(job_graph)
        # self.fitness_judgment = FitnessJudgment()

    @staticmethod
    def remove_null(a: list[str], b: list[str]) -> list[Language]:
        return [Language(name, lvl) for name, lvl in zip(a, b) if name != "-"]

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

        return DataFrame(data=dataset, dtype=np.float32).astype({"qId": "int", "kId": "int"})

    def fitness(self, offer: tuple, cv: tuple) -> dict:

        cv_lang = self.remove_null([cv[21], cv[22], cv[23]], [cv[24], cv[25], cv[26]])
        of_lang = self.remove_null([offer[22], offer[23]], [offer[25], offer[26]])
        of_comp_ess = self.filter([offer[i] for i in range(8, 11 + 1)])
        of_comp_opt = self.filter([offer[i] for i in range(12, 14 + 1)])
        of_know_ess = self.filter([offer[i] for i in range(15, 18 + 1)])
        of_know_opt = self.filter([offer[i] for i in range(19, 21 + 1)])
        cv_comp = self.filter([cv[i] for i in range(7, 13 + 1)])
        cv_know = self.filter([cv[i] for i in range(14, 20 + 1)])

        fit_edu_basic = self.fitness_edu.fitness_basic(offer[3], cv[3])
        fit_edu_bonus = self.fitness_edu.fitness_bonus(offer[4], cv[3])
        fit_exp_basic = self.fitness_experience.fitness_basic(offer[28], cv[27])
        fit_exp_bonus = self.fitness_experience.fitness_bonus(offer[28], offer[29], cv[27])
        fit_lang_basic = self.fitness_languages.fitness_basic(of_lang, cv_lang)
        fit_lang_bonus = self.fitness_languages.fitness_bonus(cv_lang, Language(offer[24], offer[27]))
        fit_competence_basic = self.fitness_skills.fitness(of_comp_ess, cv_comp)
        fit_competence_bonus = self.fitness_skills.fitness(of_comp_opt, cv_comp)
        fit_knowledge_basic = self.fitness_skills.fitness(of_know_ess, cv_know)
        fit_knowledge_bonus = self.fitness_skills.fitness(of_know_opt, cv_know)

        # fit_expertize = self.fitness_judgment.fitness_basic([fit_exp_bonus,
        #                                                     fit_competence_bonus, fit_knowledge_bonus])
        # fit_edu_judgment = self.fitness_judgment.fitness_basic([fit_edu_bonus, fit_lang_bonus])

        result = dict(
            qId=offer[0],
            kId=cv[0],
            fitness_edu_basic=fit_edu_basic,
            fitness_edu_bonus=fit_edu_bonus,
            fitness_city=self.fitness_cities.fitness(offer[7], cv[5], cv[6]),
            fitness_age=self.fitness_age.fitness_basic(cv[4], offer[5], offer[6]),
            fitness_exp_basic=fit_exp_basic,
            fitness_exp_bonus=fit_exp_bonus,
            fitness_lang_basic=fit_lang_basic,
            fitness_lang_bonus=fit_lang_bonus,
            fitness_comp_basic=fit_competence_basic,
            fitness_comp_bonus=fit_competence_bonus,
            fitness_know_basic=fit_knowledge_basic,
            fitness_knowl_bonus=fit_knowledge_bonus,
            # fit_expertize=fit_expertize,
            # fit_edu_judgment=fit_edu_judgment
        )
        return result
