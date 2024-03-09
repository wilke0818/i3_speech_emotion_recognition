from datetime import datetime

from typing import Optional, Dict, List, Union
from dataclasses import dataclass, field
from collections import namedtuple
import json

@dataclass
class ExperimentInputParameters:
    experiment_name: str #Required to allow for proper parallelism
    model_save_folder: str #Where to store final output model
    experiment_artifacts_output_path: str
    model_input_files: str #Path to the folder containing the model input jsons to run or to a specific model json if we only want to run one
    checkpoint_save_folder: Optional[str] = None #Optional pplace to save checkpoints too. Unused if checkpoint_save not specified. Will default to model_save_folder
    cross_validation: Optional[int] = None
    augmentations: Optional[Dict[str, Dict[str, Union[str, int, float]]]] = None
    datasets: Optional[List[Dict[str, str]]] = field(default_factory = lambda: []) #This is slightly off but ok for right now because we specify every time rather than use default
    hp_amount_of_training_data: Optional[float] = .3
    hp_num_trials: Optional[int] = 5
    training_dataset: Optional[Dict[str, str] ] = field(default_factory = lambda: ([{'name': 'emozionalmente', 'data_csv': './data/audio4analysis/metadata.csv'}]))
    save_steps: Optional[float] = None
    save_total_limit: Optional[int] = None
    run_eval: Optional[bool] = False
    run_grapher: Optional[bool] = False
    eval_steps: Optional[float] = 10
    logging_steps: Optional[float] = 10
    eval_metric: Optional[str] = "accuracy"

    def fromJSON(path):
        with open(path) as f:
            input_dic = json.load(f)
            return ExperimentInputParameters(**input_dic)


