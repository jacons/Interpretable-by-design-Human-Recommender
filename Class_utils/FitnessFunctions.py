import sys
from enum import Enum
from itertools import product

from pandas import read_csv

from Class_utils import JobGraph


class EducationLevel(Enum):
    A1 = 0
    A2 = 1
    B1 = 2
    B2 = 3
    C1 = 4
    C2 = 5


class FitnessFunctions:
    def __init__(self, job_graph: JobGraph, sources: dict):

        self.distance = read_csv(sources["cities_dist"], index_col=[0, 1], skipinitialspace=True)

        self.max_distance = self.distance["D"].max()

        self.education = {i[1]: i[0] for i in read_csv(sources["education_path"], index_col=0).itertuples()}
        self.lvl2value = {level.name: level.value for level in EducationLevel}
        self.education["-"] = "-"
        self.len_ed_rank = len(self.education)

        self.job_graph = job_graph

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
            bonus += 0.25 if offer_op and int(offer_ess) <= cv else 0

        return basic, bonus

    @staticmethod
    def remove_null(a: list, b: list):
        list_ = []
        for lang, lvl in zip(a, b):
            if lang != "-":
                list_.append((lang, lvl))
        return list_

    @staticmethod
    def filter(list_: list):
        return [item for item in list_ if item != "-"]

    def fitness_edu_function(self, offer_ess: str, offer_op: str, cv: str) -> tuple[float, float]:
        # max 1,25 min 0
        cv = self.education[cv]  # level of candidate's education
        offer_ess = self.education[offer_ess]  # essential education
        offer_op = self.education[offer_op]  # optional education

        basic = 1 if offer_ess <= cv else 0
        bonus = 0 if offer_op == "-" else 0.25 if offer_op <= cv else 0

        return basic, bonus

    def fitness_city_function(self, cityA: str, cityB: str, range_: int) -> float:
        # max 1 min 0
        if cityA == cityB:
            return 1

        s_cities = sorted([cityA, cityB])
        dist = self.distance.loc[(s_cities[0], s_cities[1])].values[0]

        return 1 if dist < range_ else 1 - (dist - range_) / self.max_distance

    def fitness_lange_function(self, essential: list[tuple], cv: list[tuple], optional: tuple) -> tuple[float, float]:

        basic, bonus = 0, 0
        for a, b in product(essential, cv):
            if a[0] == b[0]:
                lvl_of = self.lvl2value[a[1]]
                lvl_cv = self.lvl2value[b[1]]
                basic += 1 if lvl_of <= lvl_cv else 1 / (lvl_of - lvl_cv)

        if optional[0] != "-":
            for lang, lvl in cv:
                if lang == optional[0]:
                    lvl_of = self.lvl2value[optional[1]]
                    lvl_cv = self.lvl2value[lvl]
                    bonus += 0.25 if lvl_of <= lvl_cv else 0.25 / (lvl_of - lvl_cv)

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

    def fitness(self, offer: tuple, cv: tuple) -> dict:

        cv_lang = self.remove_null([cv[21], cv[22], cv[23]], [cv[24], cv[25], cv[26]])
        of_lang = self.remove_null([offer[21], offer[22]], [offer[24], offer[25]])
        of_comp_ess = self.filter([offer[i] for i in range(7, 10 + 1)])
        of_comp_opt = self.filter([offer[i] for i in range(11, 13 + 1)])
        of_know_ess = self.filter([offer[i] for i in range(14, 17 + 1)])
        of_know_opt = self.filter([offer[i] for i in range(18, 20 + 1)])
        cv_comp = self.filter([cv[i] for i in range(7, 13 + 1)])
        cv_know = self.filter([cv[i] for i in range(14, 20 + 1)])

        fitness_competence = self.fitness_skills_function(of_comp_ess, of_comp_opt, cv_comp)
        fitness_knowledge = self.fitness_skills_function(of_know_ess, of_know_opt, cv_know)
        fitness_edu = self.fitness_edu_function(offer[2], offer[3], cv[3])
        fitness_exp = self.fitness_experience_function(offer[27], offer[28], cv[27])
        fitness_lang = self.fitness_lange_function(of_lang, cv_lang, (offer[23], offer[26]))
        result = dict(
            qId=offer[0],
            kId=cv[0],
            fitness_edu_basic=fitness_edu[0],
            fitness_edu_bonus=fitness_edu[1],
            fitness_city=self.fitness_city_function(offer[6], cv[5], cv[6]),
            fitness_age=self.fitness_age_function(cv[4], offer[4], offer[5]),
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
