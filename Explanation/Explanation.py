import pickle

from imodels import FIGSRegressor
from pandas import DataFrame

from Class_utils.FitnessFunctions import FitnessFunctions
from Class_utils.JobGraph import JobGraph


class FIGSExplanation:
    def __init__(self, job_graph: JobGraph, fitness_functions: FitnessFunctions, model_path: str,
                 offer: DataFrame = None, curricula: DataFrame = None):
        self.job_graph = job_graph
        self.fit_functions = fitness_functions
        self.model: FIGSRegressor = self.load_model(model_path)

        self.job_offers = offer
        self.curricula = curricula

        self.current_job = None
        self.current_curricula = None

    def job_offers(self, offer: DataFrame, curricula: DataFrame):
        self.job_offers = offer
        self.curricula = curricula
        return self

    def qId(self, qId: int):
        self.current_job = self.job_offers[self.job_offers.index == qId]
        self.current_curricula = self.curricula[self.curricula["qId"] == qId]
        return self

    @staticmethod
    def load_model(name: str):
        with open(name, 'rb') as file:
            model = pickle.load(file)
        return model
