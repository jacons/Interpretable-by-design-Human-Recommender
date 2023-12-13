import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from pandas import DataFrame


class PiecewiseFunction:

    def __init__(self, model: ExplainableBoostingRegressor, idx: int):
        self.name = model.feature_names[idx]
        self.cuts = model.bins_[idx][0].tolist()
        self.contrib = model.term_scores_[idx][1:-1].tolist()
        self.std_dev = model.standard_deviations_[idx][1:-1].tolist()
        self.min_, self.max_ = model.feature_bounds_[idx]

    def __call__(self, x: float, only_output: bool = False) -> Tuple[float, float] | float:
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
        return output[0] if only_output else output

    def build_function(self, esp: float = 0.01) -> DataFrame:
        line_space = np.arange(self.min_ - 0.1, self.max_ + 0.1, esp)

        x, y_lowers, y, y_uppers = [], [], [], []

        for x_ in line_space:
            y_, y_std = self.__call__(x_)

            x.append(x_)
            y_lowers.append(y_ - y_std)
            y.append(y_)
            y_uppers.append(y_ + y_std)

        return DataFrame({"x": x, "lower": y_lowers, "y": y, "upper": y_uppers})


class ExplainableBMInterpreter:
    def __init__(self, saved_model: str) -> None:
        self.model, self.piecewise_functions, self.unique_values = self.load_model(saved_model)

        self.features_name = list(self.unique_values.keys())
        self.bias = self.model.intercept_

    @staticmethod
    def build(model: ExplainableBoostingRegressor, unique_values: dict[str, list],
              file_name_: str):
        if set(model.feature_names) != set(unique_values.keys()):
            print("Error, the features are not the same")
            return
        piecewise_functions = {f: PiecewiseFunction(model, idx) for idx, f in enumerate(model.feature_names)}

        with open(file_name_ + ".pkl", 'wb') as file:
            pickle.dump([model, piecewise_functions, unique_values], file)

        return ExplainableBMInterpreter(file_name_)

    def build_dataframe_function(self, feature_names: str) -> DataFrame:
        xs = self.unique_values[feature_names]
        y = [self.piecewise_functions[feature_names](x, True) for x in xs]

        return DataFrame({"x": xs, "y": y})

    @staticmethod
    def load_model(name: str = "model"):
        with open(name + ".pkl", 'rb') as file:
            [model, piecewise_functions, unique_values] = pickle.load(file)
        return model, piecewise_functions, unique_values

    def get_features(self):
        return self.features_name

    def get_piecewise_function(self, feature_name: str):
        return self.piecewise_functions[feature_name].build_function()

    def get_contribute(self, X: np.ndarray):
        score, contribute = self.model.predict_and_contrib(X)

        columns = [f"Candidate{i}" for i in range(contribute.shape[0])]
        contribute = pd.DataFrame(contribute.T, index=self.features_name, columns=columns)
        return score, contribute
