from datetime import datetime

from typing import Optional
from dataclasses import dataclass
from collections import namedtuple
import json

@dataclass
class ModelInputParameters:
    model_path_or_name: str
    name: Optional[str] = str(int(datetime.now().timestamp()))
    batch_size: Optional[int] = 8
    per_device_batch_size: Optional[int] = 8
    pooling_mode: Optional[str] = 'mean'
    model_output_dir: Optional[str] = './wav2vec2-xlsr'
    speaker_independent_scenario: Optional[bool] = True
    eval_steps: Optional[int] = 10
    logging_steps: Optional[int] = 10
    input_column: Optional[str] = 'path'
    output_column: Optional[str] = 'emotion'
    is_regression: Optional[bool] = False
    train_test_split: Optional[float] = .8
    seed: Optional[int] = 0
    use_batch_norm: Optional[bool] = False
    use_dropout: Optional[bool] = False
    dropout_rate: Optional[float] = .5
    use_l2_reg: Optional[bool] = False
    weight_decay: Optional[float] = .01
    number_of_training_epochs: Optional[int] = 50
    continue_model_training: Optional[bool] = False #if True, model output directory must have the last saved checkpoints of the training
    skip_hp_search: Optional[bool] = True

    def fromJSON(path):
        with open(path) as f:
            input_dic = json.load(f)
            return ModelInputParameters(**input_dic)


