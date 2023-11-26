import sys
from enum import Enum

from pandas import read_csv

from Class_utils.JobGraph import JobGraph
from Class_utils.parameters import Language, EducationLevel


class Features(Enum):
    JobRange = 0
    Age = 1
    Experience = 2
    Education = 3
    Language = 4
    Competence = 5
    Knowledge = 6


class FitnessFunction:
    max_value_basic: float = None
    max_value_bonus: float = None


class FitnessAge(FitnessFunction):
    max_value_basic = 1.0

    @staticmethod
    def fitness_basic(cv: int, v_min: int, v_max: int) -> float:
        return 1 if int(v_min <= cv <= v_max) else 0


class FitnessExperience(FitnessFunction):
    max_value_basic = 1.0
    min_value_bonus = 0.5

    @staticmethod
    def fitness_basic(offer_ess: str, cv: int) -> float:
        basic = 0
        if offer_ess != "-":
            basic += 1 if int(offer_ess) <= cv else 0
        return basic

    @staticmethod
    def fitness_bonus(offer_ess: str, offer_op: bool, cv: int) -> float:
        bonus = 0
        if offer_ess != "-":
            bonus += 0.50 if offer_op and int(offer_ess) < cv else 0
        return bonus


class FitnessEdu(FitnessFunction):
    max_value_basic = 1.0
    max_value_bonus = 0.5

    def __init__(self, education_path: str):
        # Education dictionary: "education level" -> importance. E.g. Degree-> 1
        self.education = {i[1]: i[0] for i in read_csv(education_path, index_col=0).itertuples()}

    def fitness_basic(self, offer_ess: str, cv: str) -> float:
        # max 1,50 min 0
        cv = self.education[cv]  # level of candidate's education
        offer_ess = self.education[offer_ess]  # essential education
        basic = 1 if offer_ess <= cv else 0
        return basic

    def fitness_bonus(self, offer_op: str, cv: str) -> float:
        # max 1,50 min 0
        cv = self.education[cv]  # level of candidate's education
        if offer_op != "-":
            offer_op = self.education[offer_op]  # optional education
        bonus = 0 if offer_op == "-" else 0.50 if offer_op <= cv else 0
        return bonus


class FitnessCity(FitnessFunction):
    max_value_basic = 1.0

    def __init__(self, distance_path: str):
        self.distance = read_csv(distance_path, index_col=[0, 1], skipinitialspace=True)
        self.max_distance = self.distance["Dist"].max()

    def find_distance(self, cityA: str, cityB: str):
        if cityA == cityB:
            return 1

        s_cities = sorted([cityA, cityB])
        return self.distance.loc[(s_cities[0], s_cities[1])].values[0]

    def distance_scoring(self, dist: float, range_: int):
        diff = dist - range_
        if diff <= 0:
            return 1
        score = int(diff / 20) * 260

        return max(1 - score / self.max_distance, 0)

    def fitness_basic(self, cityA: str, cityB: str, range_: int) -> float:
        dist = self.find_distance(cityA, cityB)
        return self.distance_scoring(dist, range_)


class FitnessLanguages(FitnessFunction):
    max_value_basic = 1.0
    max_value_bonus = 0.5

    lvl2value = {level.name: level.value for level in EducationLevel}

    def fitness_basic(self, essential: list[Language], cv: list[Language]) -> float:

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

        bonus = 0
        for cv_lang in cv:
            cv_level = self.lvl2value[cv_lang.level]

            if cv_lang.name == optional.name:
                bonus += 0.5 if cv_level > 0 else 0.3

        return bonus


class FitnessSkills(FitnessFunction):
    max_value_basic = 1.0
    max_value_bonus = 0.5

    #  TO COMPLETEEEEEEEEEEEEEEEEEEEEEEE
    #  TO COMPLETEEEEEEEEEEEEEEEEEEEEEEE
    #  TO COMPLETEEEEEEEEEEEEEEEEEEEEEEE
    #  TO COMPLETEEEEEEEEEEEEEEEEEEEEEEE
    #  TO COMPLETEEEEEEEEEEEEEEEEEEEEEEE
    #  TO COMPLETEEEEEEEEEEEEEEEEEEEEEEE
    #  TO COMPLETEEEEEEEEEEEEEEEEEEEEEEE
    #  TO COMPLETEEEEEEEEEEEEEEEEEEEEEEE
    def __init__(self, job_graph: JobGraph):
        self.job_graph = job_graph

    def fitness(self, essential: list, optional: list, cv: list):
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
