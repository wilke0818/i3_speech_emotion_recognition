from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
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

from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ClassificationHead(nn.Module):
    def __init__(self, config, use_dropout, dropout_rate, use_batch_norm, weight_decay):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(config.hidden_size) if use_batch_norm else nn.Identity()
        self.relu=nn.ReLU()
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.weight_decay = weight_decay

    def forward(self, features, **kwargs):
        x = features
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2) ** 2
        return self.weight_decay * l2_loss


class ModelForSpeechClassification(PreTrainedModel):
    def __init__(self, config, use_dropout=False, dropout_rate=.5, 
                 use_batch_norm=False, use_l2_reg=False, weight_decay=.01):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.config_class = config.__class__
        self.use_l2_reg = use_l2_reg
        
        self.model = AutoModel.from_pretrained(config._name_or_path)
        self.classifier = ClassificationHead(config, use_dropout, dropout_rate, use_batch_norm, weight_decay)

        self.embedding = ECAPA_TDNN(config.hidden_size, lin_neurons=config.hidden_size) 
        self.init_weights()


    def freeze_feature_extractor(self):
        #self.model.freeze_feature_extractor()
        self.model.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        elif mode == "ecapa":
            outputs = torch.squeeze(self.embedding(hidden_states))
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            if self.use_l2_reg:
                loss += self.classifier.l2_regularization_loss()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class DataCollatorCTCWithPadding:
    processor: ProcessorMixin
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        #if self.use_amp:
            #self.scaler.scale(loss).backward()
        if getattr(self, 'use_apex', None):
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif getattr(self, 'deepseed', None):
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


