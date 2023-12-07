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
        if diff <= range_ / 3:
            return 0.6
        if diff <= 2 * range_ / 3:
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
    def naive_match(offer: set, cv: set) -> tuple[set, float]:
        """
        Get a list of job-offer's skills and curriculum's skills and return the
        shared skills. In the input parameter will be removed the shared skills.
        """
        sk_shared = offer & cv
        offer -= sk_shared
        return sk_shared, len(sk_shared)

    def fitness(self, essential: list, optional: list, cv: list) -> tuple[float, float]:
        essential, optional, cv = set(essential), set(optional), set(cv)
        total_es, total_op = len(essential), len(optional)

        if self.job_graph is None:
            # ------- Score without Knowledge base -------
            es_shared = self.naive_match(essential, cv)[1]
            op_shared = self.naive_match(optional, cv)[1]

            score_es = es_shared / total_es if total_es > 0 else 0
            score_op = op_shared / total_op if total_op > 0 else 0
            # ------- Score without Knowledge base -------
        else:
            # ------- Score with Knowledge base -------
            perfect_es_shared, n_per_es_shared = self.naive_match(essential, cv)
            perfect_op_shared, n_per_op_shared = self.naive_match(optional, cv)

            # ----------- standardize cv -----------
            unique_cv, amb_cv = self.job_graph.skill_standardize(cv)
            amb_cv = self.job_graph.solve_ambiguous(amb_cv, unique_cv)
            cv = unique_cv + amb_cv
            # ----------- standardize cv -----------

            unique_es_uri, amb_es_uri = self.job_graph.skill_standardize(essential)
            perfect_es_shared, _ = self.job_graph.skill_standardize(perfect_es_shared)

            unique_op_uri, amb_op_uri = self.job_graph.skill_standardize(optional)
            perfect_op_shared, _ = self.job_graph.skill_standardize(perfect_op_shared)

            contex = unique_es_uri + unique_op_uri + perfect_es_shared + perfect_op_shared

            amb_es = self.job_graph.solve_ambiguous(amb_es_uri, contex)
            essential = unique_es_uri + amb_es

            amb_op = self.job_graph.solve_ambiguous(amb_op_uri, contex)
            optional = unique_op_uri + amb_op

            essential, optional, cv = set(essential), set(optional), set(cv)
            imperfect_es_shared, n_imper_es_shared = self.naive_match(essential, cv)
            imperfect_op_shared, n_imper_op_shared = self.naive_match(optional, cv)

            # with the remain skill in the both lists, we apply the similarity score
            sim_score_es = mean(self.job_graph.node_similarity(essential, cv, ids=True))
            sim_score_op = mean(self.job_graph.node_similarity(optional, cv, ids=True))

            es_shared = n_per_es_shared + n_imper_es_shared
            op_shared = n_per_op_shared + n_imper_op_shared
            sub_score_es = es_shared / total_es if total_es > 0 else 0
            sub_score_op = op_shared / total_op if total_op > 0 else 0

            score_es = sub_score_es + (1 - sub_score_es) * sim_score_es
            score_op = sub_score_op + (1 - sub_score_op) * sim_score_op
            # ------- Score with Knowledge base -------
        return score_es, score_op


class FitnessJudgment:

    @staticmethod
    def fitness_basic(features: list[float]) -> float:
        return np.log1p(np.log1p(np.power(sum(features), 3)))
