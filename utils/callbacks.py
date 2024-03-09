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
from typing import Any, Dict, Union, Optional


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



class CustomizedEarlyStoppingCallback(TrainerCallback):
  def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0, epochs_to_see: Optional[float] = 1.0):
    self.early_stopping_patience = early_stopping_patience
    self.early_stopping_threshold = early_stopping_threshold
    self.epochs_to_see = epochs_to_see
    # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
    self.early_stopping_patience_counter = 0

  def check_metric_value(self, args, state, control, metric_value):
    # best_metric is set by code for load_best_model
    operator = np.greater if args.greater_is_better else np.less
    if state.best_metric is None or (
      operator(metric_value, state.best_metric) and abs(metric_value - state.best_metric) > self.early_stopping_threshold
    ):
      self.early_stopping_patience_counter = 0
    else:
      self.early_stopping_patience_counter += 1

  def on_train_begin(self, args, state, control, **kwargs):
    assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
    assert args.metric_for_best_model is not None, "EarlyStoppingCallback requires metric_for_best_model is defined"

  def on_evaluate(self, args, state, control, metrics, **kwargs):
    metric_to_check = args.metric_for_best_model
    if not metric_to_check.startswith("eval_"):
      metric_to_check = f"eval_{metric_to_check}"
    metric_value = metrics.get(metric_to_check)

    if metric_value is None:
      print(f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping is disabled")
      return

    self.check_metric_value(args, state, control, metric_value)
    if self.early_stopping_patience_counter >= self.early_stopping_patience and state.epoch >= self.epochs_to_see:
      control.should_training_stop = True


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


