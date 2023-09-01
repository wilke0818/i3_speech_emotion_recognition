import seaborn as sn
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
import sys
from collections import OrderedDict
import IPython.display as ipd
from sklearn.model_selection import GroupShuffleSplit
from datasets import load_dataset, load_metric
from transformers import AutoConfig, Wav2Vec2Processor, AutoModel
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import transformers
from transformers import Wav2Vec2Processor
from transformers import EvalPrediction
from transformers import TrainingArguments
from transformers import PreTrainedModel
from transformers import ProcessorMixin
from transformers import TrainerCallback
from transformers import TrainingArguments, AutoModelForCTC
from typing import Any, Dict, Union


import torch
from packaging import version
from torch import nn

from transformers import (
    Trainer,
    is_apex_available,
)
import optuna
from optuna.trial import TrialState
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.metrics import classification_report

import shutil
import os
import json


#TODO organize/cleanup these callbacks
class PrinterCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_local_process_zero:
            with open(args.logging_dir+"/logs.txt", "a") as f:
                f.write(json.dumps(metrics)+'\n')
                print(str(metrics))

class LoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            for name, value in state.log_history[-1].items():
                print(f"{name} at step {state.global_step}: {value}")
                

class MemorySaverCallback(TrainerCallback):
    
    def __init__(self):
        super(MemorySaverCallback, self).__init__()
        self.run_num = 0

    def on_train_begin(self, args, state, control, **kwargs):
        print(f'attempting to remove models: on run {self.run_num}')
        if state.is_hyper_param_search:
            files = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, f)) and f != 'runs']
            for file in files:
                shutil.rmtree(file)

    def on_train_end(self, args, state, control, **kwargs):
        self.run_num+=1


