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
            return 0.5
        else:
            return 0

    def fitness(self, cityA: str, cityB: str, range_: int) -> float:
        # max 1 min 0
        dist = self.find_distance(cityA, cityB)
        return self.distance_scoring(dist, range_)


class FitnessLanguages:
    lvl2value = {level.name: level.value for level in EducationLevel}

    def fitness_basic(self, essential: list[Language], cv: list[Language]) -> tuple[float, float]:
        basic_language, basic_level = 0, 0

        for cv_lang in cv:
            cv_level = self.lvl2value[cv_lang.level]

            for ess_lang in essential:
                if cv_lang.name == ess_lang.name:
                    basic_language += 1
                    ess_level = self.lvl2value[ess_lang.level]
                    basic_level += 1 if ess_level <= cv_level else 0

        score_language = basic_language / len(essential)
        score_level = basic_level / len(essential)
        return score_language, score_level

    def fitness_bonus(self, cv: list[Language], optional: Language) -> float:
        # max 1 min 0
        bonus = 0
        for cv_lang in cv:
            cv_level = self.lvl2value[cv_lang.level]
            if cv_lang.name == optional.name:
                bonus += 1 if cv_level > 0 else 0.5
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

    def standardize_offer(self, essential: set, optional: set, es_shared: set,
                          op_shared: set) -> tuple[set, set]:

        unique_es_uri, amb_es_uri = self.job_graph.skill_standardize(essential)
        es_shared, _ = self.job_graph.skill_standardize(es_shared)

        unique_op_uri, amb_op_uri = self.job_graph.skill_standardize(optional)
        op_shared, _ = self.job_graph.skill_standardize(op_shared)

        contex = unique_es_uri + unique_op_uri + es_shared + op_shared

        amb_es = self.job_graph.solve_ambiguous(amb_es_uri, contex_uri=contex)
        essential = set(unique_es_uri + amb_es)

        amb_op = self.job_graph.solve_ambiguous(amb_op_uri, contex_uri=contex)
        optional = set(unique_op_uri + amb_op)

        return essential, optional

    def similarity_score(self, offer: set, cv: set) -> float:
        o = mean(self.job_graph.node_similarity(offer, cv, ids=True))
        return self.discretize_similarity(o)

    @staticmethod
    def discretize_similarity(number: float) -> float:
        if number <= 0:
            return 0
        elif 0 < number < 0.25:
            return 0.25
        elif 0.25 <= number < 0.50:
            return 0.50
        elif 0.5 <= number < 0.75:
            return 0.75
        elif 0.75 <= number:
            return 1

    def fitness(self, essential: list, optional: list, cv: list) -> tuple[tuple[float, float], tuple[float, float]]:
        essential, optional, cv = set(essential), set(optional), set(cv)
        total_es, total_op = len(essential), len(optional)

        perfect_es_shared, es_shared = self.naive_match(essential, cv)
        perfect_op_shared, op_shared = self.naive_match(optional, cv)

        sim_score_es, sim_score_op = 0, 0
        if self.job_graph is not None:
            # ------- Score with Knowledge base -------
            # ----------- standardize cv -----------
            unique_cv, amb_cv = self.job_graph.skill_standardize(cv)
            amb_cv = self.job_graph.solve_ambiguous(amb_cv, contex_uri=unique_cv)
            cv = set(unique_cv + amb_cv)
            # ----------- standardize cv -----------

            # ----------- standardize essential/optional -----------
            essential, optional = self.standardize_offer(essential, optional, perfect_es_shared, perfect_op_shared)
            # ----------- standardize essential/optional -----------

            imperfect_es_shared, n_imper_es_shared = self.naive_match(essential, cv)
            imperfect_op_shared, n_imper_op_shared = self.naive_match(optional, cv)

            # with the remain skill in the both lists, we apply the similarity score
            sim_score_es = self.similarity_score(essential, cv)
            sim_score_op = self.similarity_score(optional, cv)

            es_shared += n_imper_es_shared
            op_shared += n_imper_op_shared
            # ------- Score with Knowledge base -------

        score_es = es_shared / total_es if total_es > 0 else 0
        score_op = op_shared / total_op if total_op > 0 else 0
        return (score_es, sim_score_es), (score_op, sim_score_op)

    def debug_score(self, essential: list, optional: list, cv: list):
        essential, optional, cv = set(essential), set(optional), set(cv)
        total_es, total_op = len(essential), len(optional)

        perfect_es_shared, es_shared = self.naive_match(essential, cv)
        perfect_op_shared, op_shared = self.naive_match(optional, cv)

        print("The shared essential skills are:", perfect_es_shared)
        print("The shared optional skills are:", perfect_op_shared)

        print("Skill for cv", cv)
        print("Remaining essential skill for job", essential)
        print("Remaining optional skill for job", optional)

        # ----------- standardize cv -----------
        unique_cv, amb_cv = self.job_graph.skill_standardize(cv)
        amb_cv = self.job_graph.solve_ambiguous(amb_cv, contex_uri=unique_cv)
        cv = set(unique_cv + amb_cv)
        # ----------- standardize cv -----------
        print("Standardized Skill for cv", cv)

        # ----------- standardize essential/optional -----------
        essential, optional = self.standardize_offer(essential, optional, perfect_es_shared, perfect_op_shared)
        # ----------- standardize essential/optional -----------
        print("Standardized essential skill", essential)
        print("Standardized optional skill", optional)
        imperfect_es_shared, n_imper_es_shared = self.naive_match(essential, cv)
        imperfect_op_shared, n_imper_op_shared = self.naive_match(optional, cv)

        print("The shared essential skills:", imperfect_es_shared)
        print("The shared optional skills:", imperfect_op_shared)

        print("Remaining essential skill", essential)
        print("Remaining optional skill", optional)

        # with the remain skill in the both lists, we apply the similarity score
        sim_score_es = self.similarity_score(essential, cv)
        sim_score_op = self.similarity_score(optional, cv)

        print("Similarity essential skill for job", sim_score_es)
        print("Similarity optional skill for job", sim_score_op)


class FitnessJudgment:

    @staticmethod
    def fitness_basic(features: list[float]) -> float:
        return np.log1p(np.log1p(np.power(sum(features), 3)))
