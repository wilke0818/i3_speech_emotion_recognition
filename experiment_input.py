from datetime import datetime

from typing import Optional, Dict, List, Union
from dataclasses import dataclass, field
from collections import namedtuple
import json

@dataclass
class ExperimentInputParameters:
    experiment_name: str #Required to allow for proper parallelism
    output_path: str #Where to store final output model
    model_files: str #Path to the folder containing the model input jsons to run or to a specific model json if we only want to run one
    cross_validation: Optional[int] = None
    augmentations: Optional[Dict[str, Dict[str, Union[str, int, float]]]] = None
    datasets: Optional[List[Dict[str, str]]] = field(default_factory = lambda: ([{'name': 'emozionalmente'}]))
    hp_amount_of_training_data: Optional[float] = .3
    hp_num_trials: Optional[int] = 5
    training_data_path: Optional[str] = './data/audio4analysis/'

    def fromJSON(path):
        with open(path) as f:
            input_dic = json.load(f)
            return ExperimentInputParameters(**input_dic)


