import json
import os
import pickle
from typing import Tuple

from lightgbm import LGBMRanker


class GridSearch:
    @staticmethod
    def save_model_parameters(model: LGBMRanker, parameter: dict, name: str = "model"):
        if not os.path.exists("model"):
            os.makedirs("model")

        with open('model/'+name+".pkl", 'wb') as file:
            pickle.dump(model, file)

        with open("model/"+name+"_best_parameter.json", 'w') as file:
            json.dump(parameter, file)

    @staticmethod
    def load_model_parameters(name: str = "model") -> Tuple[LGBMRanker, dict]:
        with open('model/'+name+".pkl", 'rb') as file:
            model = pickle.load(file)
        with open("model/"+name+"_best_parameter.json", 'r') as file:
            parameter = json.load(file)

        return model, parameter
