from abc import abstractmethod

from pandas import DataFrame, read_csv


class FitnessFunction:
    max_value: float = None
    min_value: float = None


class FitnessAge(FitnessFunction):
    max_value = 1.0
    min_value = 0.0

    @staticmethod
    def fitness(cv: int, v_min: int, v_max: int) -> float:
        return 1 if int(v_min <= cv <= v_max) else 0


class FitnessExperience(FitnessFunction):
    max_value = 1.5
    min_value = 0.0

    @staticmethod
    def fitness(offer_ess: str, offer_op: bool, cv: int) -> tuple[float, float]:
        basic, bonus = 0, 0
        if offer_ess != "-":
            basic += 1 if int(offer_ess) <= cv else 0
            bonus += 0.50 if offer_op and int(offer_ess) < cv else 0

        return basic, bonus


class FitnessEdu(FitnessFunction):
    max_value = 1.5
    min_value = 0.0

    def __init__(self, education_path: str):
        # Education dictionary: "education level" -> importance. E.g. Degree-> 1
        self.education = {i[1]: i[0] for i in read_csv(education_path, index_col=0).itertuples()}

    def fitness(self, offer_ess: str, offer_op: str, cv: str) -> tuple[float, float]:
        # max 1,50 min 0
        cv = self.education[cv]  # level of candidate's education
        offer_ess = self.education[offer_ess]  # essential education

        if offer_op != "-":
            offer_op = self.education[offer_op]  # optional education

        basic = 1 if offer_ess <= cv else 0
        bonus = 0 if offer_op == "-" else 0.50 if offer_op <= cv else 0

        return basic, bonus


class FitnessCity(FitnessFunction):
    max_value = 1.0
    min_value = 0.0

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

    def fitness(self, cityA: str, cityB: str, range_: int) -> float:
        dist = self.find_distance(cityA, cityB)
        return self.distance_scoring(dist, range_)
