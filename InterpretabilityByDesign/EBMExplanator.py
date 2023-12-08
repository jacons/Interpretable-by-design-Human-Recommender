import pickle
from typing import Tuple

import numpy as np
from pandas import DataFrame


class EBMExplanator:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

        self.piecewise_functions = dict()
        self.build_piecewise_functions()

    @staticmethod
    def load_model(name: str = "model"):
        with open(name + ".pkl", 'rb') as file:
            model = pickle.load(file)
        return model

    def build_piecewise_functions(self):
        for idx, feature in enumerate(self.model.feature_names):
            min_, max_ = self.model.feature_bounds_[idx]
            fun = PiecewiseFunction(feature,
                                    self.model.bins_[idx][0],
                                    self.model.term_scores_[idx][1:-1],
                                    self.model.standard_deviations_[idx][1:-1],
                                    min_, max_)
            self.piecewise_functions[feature] = fun

    def show_piecewise_functions(self):
        # return [i.show_function() for i in self.piecewise_functions.values()]
        return [(name, fun.show_function()) for name, fun in self.piecewise_functions.items()]


class PiecewiseFunction:
    def __init__(self, name: str, cuts: np.ndarray, contrib: np.ndarray, std_dev: np.ndarray,
                 min_: float, max_: float):
        self.name = name
        self.cuts = cuts.tolist()
        self.contrib = contrib.tolist()
        self.std_dev = std_dev.tolist()
        self.min_ = min_
        self.max_ = max_

    def get_result(self, x: float) -> Tuple[float, float]:
        output = None

        if x <= self.cuts[0]:
            output = self.contrib[0], self.std_dev[0]
        elif x >= self.cuts[-1]:
            output = self.contrib[-1], self.std_dev[-1]
        else:
            for i in range(len(self.cuts) - 1):
                if self.cuts[i] <= x <= self.cuts[i + 1]:
                    output = self.contrib[i + 1], self.std_dev[i + 1]
                    break

        return output

    def show_function(self) -> DataFrame:
        line_space = np.arange(self.min_, self.max_, 0.01)

        x, y_lowers, y, y_uppers = [], [], [], []

        for x_ in line_space:
            y_, y_std = self.get_result(x_)

            x.append(x_)
            y_lowers.append(y_ - y_std)
            y.append(y_)
            y_uppers.append(y_ + y_std)

        return DataFrame({"x": x, "lower": y_lowers, "y": y, "upper": y_uppers})
