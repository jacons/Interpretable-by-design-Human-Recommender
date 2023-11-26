import pickle
from enum import Enum

from imodels import FIGSRegressor
from pandas import DataFrame

from Class_utils.FitnessClasses import Features
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
        self.fitness_matrix = None

    def set_offers_curricula(self, offer: DataFrame, curricula: DataFrame, qId: int = None):
        self.job_offers = offer
        self.curricula = curricula
        if qId is not None:
            self.qId(qId)
        return self

    def qId(self, qId: int):
        if self.job_offers is None or self.curricula is None:
            print("Job-offers and Curricula are null")
            return

        self.current_job = self.job_offers[self.job_offers.index == qId]
        self.current_curricula = self.curricula[self.curricula.index.get_level_values(0) == qId]
        return self

    @staticmethod
    def load_model(name: str):
        with open(name, 'rb') as file:
            model = pickle.load(file)
        return model

    def perform_top(self):
        if self.current_job is None or self.current_curricula is None:
            print("Current job-offer and Current-curricula are Null")
            return

        fitness_matrix = self.fit_functions.generate_fitness_score(self.current_job,
                                                                   self.current_curricula)
        fitness_matrix["lambda"] = self.model.predict(fitness_matrix.iloc[:, 2:].to_numpy())
        fitness_matrix.sort_values("lambda", inplace=True, ascending=False)
        self.fitness_matrix = fitness_matrix.reset_index()

    def get_position(self, kId: int) -> int:
        return self.fitness_matrix[self.fitness_matrix.kId == kId].index.values[0]

    def get_max_position(self, kId: int, exclude: list[Features] = None):
        if exclude is None:
            exclude = []
        fitness_vector = self.fitness_matrix[self.fitness_matrix.kId == kId]

        idea_vector = []
        for k, v in self.fit_functions.fitness_max_values.items():
            pass
