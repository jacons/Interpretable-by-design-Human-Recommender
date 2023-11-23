import sys

import numpy as np
from pandas import read_csv, DataFrame
from tqdm import tqdm

from Class_utils import JobGraph
from Class_utils.parameters import EducationLevel, Language


class FitnessFunctions:
    def __init__(self, job_graph: JobGraph, sources: dict):

        # ------------------------ LOAD RESOURCES ------------------------
        # --- Cities ---
        self.distance = read_csv(sources["cities_dist"], index_col=[0, 1], skipinitialspace=True)
        self.max_distance = self.distance["Dist"].max()
        # --- Cities ---

        # --- Education ---
        # Education dictionary: "education level" -> importance. E.g. Degree-> 1
        self.education = {i[1]: i[0] for i in read_csv(sources["education_path"], index_col=0).itertuples()}
        self.len_ed_rank = len(self.education)
        # --- Education ---

        # --- Language and language levels ---
        self.lvl2value = {level.name: level.value for level in EducationLevel}
        # --- Language and language levels ---

        # --- Skills and Occupations ---
        self.job_graph = job_graph
        # --- Skills and Occupations ---
        # ------------------------ LOAD RESOURCES ------------------------
        return

    @staticmethod
    def fitness_age_function(cv: int, v_min: int, v_max: int) -> float:
        # max 1 min 0
        return 1 if int(v_min <= cv <= v_max) else 0

    @staticmethod
    def fitness_experience_function(offer_ess: str, offer_op: bool, cv: int) -> tuple[float, float]:
        # max 1,25 min 0
        basic, bonus = 0, 0
        if offer_ess != "-":
            basic += 1 if int(offer_ess) <= cv else 0
            bonus += 0.50 if offer_op and int(offer_ess) < cv else 0

        return basic, bonus

    def fitness_edu_function(self, offer_ess: str, offer_op: str, cv: str) -> tuple[float, float]:
        # max 1,50 min 0
        cv = self.education[cv]  # level of candidate's education
        offer_ess = self.education[offer_ess]  # essential education

        if offer_op != "-":
            offer_op = self.education[offer_op]  # optional education

        basic = 1 if offer_ess <= cv else 0
        bonus = 0 if offer_op == "-" else 0.50 if offer_op <= cv else 0

        return basic, bonus

    def fitness_city_function(self, cityA: str, cityB: str, range_: int) -> float:
        # max 1 min 0
        if cityA == cityB:
            return 1

        s_cities = sorted([cityA, cityB])
        dist = self.distance.loc[(s_cities[0], s_cities[1])].values[0]

        return 1 if dist < range_ else 1 - (dist - range_) / self.max_distance

    def fitness_lange_function(self, essential: list[Language], cv: list[Language],
                               optional: Language) -> tuple[float, float]:

        basic, bonus = 0, 0
        for cv_lang in cv:
            cv_level = self.lvl2value[cv_lang.level]

            for ess_lang in essential:
                if cv_lang.name == ess_lang.name:
                    if ess_lang.level == "Any":
                        basic += 1 if cv_level > 0 else 0.7
                    else:
                        jo_level = self.lvl2value[ess_lang.level]
                        basic += 1 if jo_level <= cv_level else 1 / (2 * (jo_level - cv_level))

            if cv_lang.name == optional.name:
                bonus += 0.5 if cv_level > 0 else 0.3

        return basic / len(essential), bonus

    def fitness_skills_function(self, essential: list, optional: list, cv: list):
        job_graph = self.job_graph

        basic, bonus, min_distance = 0, 0, sys.maxsize
        essential, optional, cv = set(essential), set(optional), set(cv)

        # ------- Score without Knowledge base -------
        sk_shared_es = essential & cv
        if len(essential) > 0:
            basic += 1 * len(sk_shared_es) / len(essential)

        sk_shared_op = optional & cv
        if len(optional) > 0:
            bonus += 0.5 * (len(sk_shared_op) / len(optional))
        # ------- Score without Knowledge base -------

        # ------- Score with Knowledge base (ALGO1)-------
        # id_occ = job_graph.name2id[occupation]
        # cv -= sk_shared_es
        #
        # for occ in job_graph.get_job_with_skill(cv):
        #     dist_ = job_graph.graph.edges[id_occ, occ]["weight"]
        #     min_distance = min_distance if dist_ > min_distance else dist_
        # bonus += 1 / min_distance
        # ------- Score with Knowledge base (ALGO1)-------

        # ------- Score with Knowledge base (ALGO2)-------
        if len(essential) > 0:
            essential -= sk_shared_es
            basic += 0.5 * job_graph.node_similarity(essential, cv - sk_shared_es, ids=False)

        if len(optional) > 0:
            optional -= sk_shared_op
            bonus += 0.25 * job_graph.node_similarity(optional, cv - sk_shared_op, ids=False)
        # ------- Score with Knowledge base (ALGO2)-------

        return basic, bonus

    @staticmethod
    def remove_null(a: list[str], b: list[str]) -> list[Language]:
        list_ = []
        for name, lvl in zip(a, b):
            if name != "-":
                list_.append(Language(name, lvl))
        return list_

    @staticmethod
    def filter(list_: list):
        return [item for item in list_ if item != "-"]

    def generate_fitness_score(self, offers: DataFrame, curricula: DataFrame) -> DataFrame:
        dataset = []
        bar = tqdm(offers.itertuples(), total=len(offers), desc="Generating the fitness scores")
        for offer in bar:
            curricula_ = curricula[curricula.index.get_level_values(0) == offer[0]]
            for cv in curricula_.itertuples():
                dataset.append(self.fitness(offer, cv))
            bar.set_postfix(qId=offer[0])

        dataset = DataFrame(data=dataset, dtype=np.float32).astype({"qId": "int", "kId": "int"})
        return dataset

    def fitness(self, offer: tuple, cv: tuple) -> dict:

        cv_lang = self.remove_null([cv[20], cv[21], cv[22]], [cv[23], cv[24], cv[25]])
        of_lang = self.remove_null([offer[22], offer[23]], [offer[25], offer[26]])
        of_comp_ess = self.filter([offer[i] for i in range(8, 11 + 1)])
        of_comp_opt = self.filter([offer[i] for i in range(12, 14 + 1)])
        of_know_ess = self.filter([offer[i] for i in range(15, 18 + 1)])
        of_know_opt = self.filter([offer[i] for i in range(19, 21 + 1)])
        cv_comp = self.filter([cv[i] for i in range(6, 12 + 1)])
        cv_know = self.filter([cv[i] for i in range(13, 19 + 1)])

        fitness_competence = self.fitness_skills_function(of_comp_ess, of_comp_opt, cv_comp)
        fitness_knowledge = self.fitness_skills_function(of_know_ess, of_know_opt, cv_know)
        fitness_edu = self.fitness_edu_function(offer[3], offer[4], cv[2])
        fitness_exp = self.fitness_experience_function(offer[28], offer[29], cv[26])
        fitness_lang = self.fitness_lange_function(of_lang, cv_lang, Language(offer[24], offer[27]))

        result = dict(
            qId=offer[0],
            kId=cv[0][1],
            fitness_edu_basic=fitness_edu[0],
            fitness_edu_bonus=fitness_edu[1],
            fitness_city=self.fitness_city_function(offer[7], cv[4], cv[5]),
            fitness_age=self.fitness_age_function(cv[3], offer[5], offer[6]),
            fitness_exp_basic=fitness_exp[0],
            fitness_exp_bonus=fitness_exp[1],
            fitness_lang_basic=fitness_lang[0],
            fitness_lang_bonus=fitness_lang[1],
            fitness_comp_basic=fitness_competence[0],
            fitness_comp_bonus=fitness_competence[1],
            fitness_know_basic=fitness_knowledge[0],
            fitness_knowl_bonus=fitness_knowledge[1],
        )
        return result
