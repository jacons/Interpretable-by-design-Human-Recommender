from typing import Literal, Tuple

import numpy as np
from numpy import mean
from pandas import read_csv

from Class_utils.JobGraph import JobGraph
from Class_utils.parameters import Language, EducationLevel


class FitnessAge:
    max_value_basic = 1.0

    @staticmethod
    def fitness_basic(cv: int, v_min: int, v_max: int) -> float:
        # max 1 min 0
        return 1 if int(v_min <= cv <= v_max) else 0


class FitnessExperience:

    @staticmethod
    def fitness_basic(offer_ess: str, cv: int) -> float:
        # max 1 min 0
        basic = 0
        if offer_ess != "-":
            basic += 1 if int(offer_ess) <= cv else 0
        return basic

    @staticmethod
    def fitness_bonus(offer_ess: str, offer_op: bool, cv: int) -> float:
        # max 1 min 0
        bonus = 0
        if offer_ess != "-":
            bonus += 1 if offer_op and int(offer_ess) < cv else 0
        return bonus


class FitnessEdu:

    def __init__(self, education_path: str):
        # Education dictionary: "education level" -> importance. E.g. Degree-> 1
        self.education = {i[1]: i[0] for i in read_csv(education_path, index_col=0).itertuples()}

    def fitness_basic(self, offer_ess: str, cv: str) -> float:
        # max 1 min 0
        cv = self.education[cv]  # level of candidate's education
        offer_ess = self.education[offer_ess]  # essential education
        basic = 1 if offer_ess <= cv else 0
        return basic

    def fitness_bonus(self, offer_op: str, cv: str) -> float:
        # max 1 min 0
        cv = self.education[cv]  # level of candidate's education
        if offer_op != "-":
            offer_op = self.education[offer_op]  # optional education
        bonus = 0 if offer_op == "-" else 1 if offer_op <= cv else 0
        return bonus


class FitnessCity:
    def __init__(self, distance_path: str):
        self.distance = read_csv(distance_path, index_col=[0, 1], skipinitialspace=True)

    def find_distance(self, cityA: str, cityB: str):
        if cityA == cityB:
            return 1

        s_cities = sorted([cityA, cityB])
        return self.distance.loc[(s_cities[0], s_cities[1])].values[0]

    @staticmethod
    def distance_scoring(dist: float, range_: int):
        diff = dist - range_

        if diff <= 0:
            return 1
        if diff <= range_/3:
            return 0.6
        if diff <= 2*range_/3:
            return 0.3
        else:
            return 0

    def fitness(self, cityA: str, cityB: str, range_: int) -> float:
        # max 1 min 0
        dist = self.find_distance(cityA, cityB)
        return self.distance_scoring(dist, range_)


class FitnessLanguages:
    lvl2value = {level.name: level.value for level in EducationLevel}

    def fitness_basic(self, essential: list[Language], cv: list[Language]) -> float:
        # max 1 min 0
        basic = 0
        for cv_lang in cv:
            cv_level = self.lvl2value[cv_lang.level]

            for ess_lang in essential:
                if cv_lang.name == ess_lang.name:
                    if ess_lang.level == "Any":
                        basic += 1 if cv_level > 0 else 0.7
                    else:
                        jo_level = self.lvl2value[ess_lang.level]
                        basic += 1 if jo_level <= cv_level else 1 / (2 * (jo_level - cv_level))

        return basic / len(essential)

    def fitness_bonus(self, cv: list[Language], optional: Language) -> float:
        # max 1 min 0
        bonus = 0
        for cv_lang in cv:
            cv_level = self.lvl2value[cv_lang.level]

            if cv_lang.name == optional.name:
                bonus += 1 if cv_level > 0 else 0.7

        return bonus


class FitnessSkills:

    def __init__(self, job_graph: JobGraph = None):
        self.job_graph = job_graph

    @staticmethod
    def naive_match(offer: set, cv: set) -> set:
        """
        Get a list of job-offer's skills and curriculum's skills and return the
        shared skills. In the input parameter will be removed the shared skills.
        """
        sk_shared = offer & cv
        offer -= sk_shared
        cv -= sk_shared

        return sk_shared

    def graph_score(self, offer: set, cv: set) -> Tuple[int, float]:
        """
        Get a list of job-offer's skills and curriculum's skills and return the
        shared skills and the simirity score with the remaining.
        In the input parameter will be removed the shared skills.
        :param offer:
        :param cv:
        :return:
        """

        # Make the "name" of skill invariant respect to their synonyms
        offer_uri = self.job_graph.skill_standardize(offer)
        cv_uri = self.job_graph.skill_standardize(cv)
        # then try again to do the "perfect" match (after mapped all synonyms)
        imperfect_shared = len(self.naive_match(offer_uri, cv_uri))

        # with the remain skill in the both lists, we apply the similarity score
        sim_score = mean(self.job_graph.node_similarity(offer_uri, cv_uri, ids=True))

        return imperfect_shared, sim_score

    def fitness(self, essential: list, cv: list) -> float:
        essential, cv = set(essential), set(cv)

        total = len(essential)
        if total <= 0:
            return 0

        # ------- Score without Knowledge base -------
        perfect_shared = len(self.naive_match(essential, cv))
        # ------- Score without Knowledge base -------

        shared = perfect_shared
        if self.job_graph is not None and len(essential) > 0:
            # ------- Score with Knowledge base -------
            imperfect_shared, sim_score = self.graph_score(essential, cv)
            # discretize the sim_score? better explainability?
            shared += imperfect_shared
            sub_score = shared / total
            score = sub_score + (1-sub_score) * sim_score
            # ------- Score with Knowledge base -------
        else:
            score = shared / total
        return score

    def debug_score(self, essential: list, cv: list):
        essential, cv = set(essential), set(cv)

        total = len(essential)
        if total <= 0:
            return 0
        # ------- Score without Knowledge base -------
        perfect_shared = self.naive_match(essential, cv)
        # ------- Score without Knowledge base -------
        print("The shared skills are:", perfect_shared)

        print("Remaining skill for cv", cv)
        print("Remaining skill for job", essential)

        shared = len(perfect_shared)
        if self.job_graph is not None and len(essential) > 0:
            # ------- Score with Knowledge base -------

            # Make the "name" of skill invariant respect to their synonyms
            offer_uri = self.job_graph.skill_standardize(essential)
            cv_uri = self.job_graph.skill_standardize(cv)
            print("The standardize skill for cv", cv_uri)
            print("The standardize skill for job", offer_uri)

            # then try again to do the "perfect" match (after mapped all synonyms)
            imperfect_shared = self.naive_match(offer_uri, cv_uri)
            print("The shared skills are:", imperfect_shared)

            # with the remain skill in the both lists, we apply the similarity score
            sim_score = mean(self.job_graph.node_similarity(offer_uri, cv_uri, ids=True))
            print("The similarity score is ", sim_score)

            shared += len(imperfect_shared)
            max_sim_score = 1 - (shared / total)
            score = shared / total + max_sim_score * sim_score
            # ------- Score with Knowledge base -------
        else:
            score = shared / total
        print("The final score is", score)


class FitnessJudgment:

    @staticmethod
    def fitness_basic(features: list[float]) -> float:
        return np.log1p(np.log1p(np.power(sum(features), 3)))
