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

from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, AttentiveStatisticsPooling, Conv1d, BatchNorm1d


# Code modeled after ECAPA_TDNN using onlt AttentiveStatisticsPooling and later:
# https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/lobes/models/ECAPA_TDNN.html#ECAPA_TDNN
class AttStats(torch.nn.Module):
  def __init__(
    self,
    input_size,
    lin_neurons=192,
    attention_channels=128,
    global_context=True
  ):
    super().__init__()
    # Attentive Statistical Pooling
    self.asp = AttentiveStatisticsPooling(
      input_size,
      attention_channels=attention_channels,
      global_context=global_context,
    )
    self.asp_bn = BatchNorm1d(input_size=input_size * 2)

    # Final linear transformation
    self.fc = Conv1d(
      in_channels=input_size * 2,
      out_channels=lin_neurons,
      kernel_size=1,
    )

  def forward(self, x, lengths=None):
      """Returns the embedding vector.

       Arguments
      ---------
      x : torch.Tensor
          Tensor of shape (batch, time, channel).
      """
      # Minimize transpose for efficiency
      x = x.transpose(1, 2)

      # Attentive Statistical Pooling
      x = self.asp(x, lengths=lengths)
      x = self.asp_bn(x)

      # Final linear transformation
      x = self.fc(x)

      x = x.transpose(1, 2)
      return x


class LinearWeightedAvg(nn.Module):
    def __init__(self, config, n_inputs):
        super().__init__()
        self.config = config
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(1)) for i in range(n_inputs)])

    def forward(self, input):
        res = 0
        for emb_idx, emb in enumerate(input):
            res += emb * self.weights[emb_idx]
        return res


def CCCLoss(x, y):
    
    ccc = 2*torch.cov(torch.stack((x,y)))[0][1] / (x.var() + y.var() + (x.mean() - y.mean())**2)
    return ccc

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout_rate) if config.use_dropout else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(config.hidden_size) if config.use_batch_norm else nn.Identity()
        self.relu=nn.ReLU() if config.use_relu else nn.Identity()
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.weight_decay = config.weight_decay

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
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.config_class = config.__class__
        self.use_l2_reg = config.use_l2_reg
        self.weight_encoder_layers = LinearWeightedAvg(config, config.num_hidden_layers+1) if config.use_weight_encoder_layers else None
        self.pool_position = config.pool_position
        
        self.pretrained_model = AutoModel.from_pretrained(config._name_or_path)
        self.custom_classifier = ClassificationHead(config)
        self.ecapa_embedding = ECAPA_TDNN(config.hidden_size, lin_neurons=config.hidden_size) if config.pooling_mode =='ecapa' else None 
        self.att_stats = AttStats(config.hidden_size, lin_neurons=config.hidden_size) if config.pooling_mode =='att_stats' else None
        self.init_weights()


    def freeze_feature_extractor(self):
        #self.model.freeze_feature_extractor()
        self.pretrained_model.feature_extractor._freeze_parameters()

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
            outputs = torch.squeeze(self.ecapa_embedding(hidden_states))
        elif mode == "att_stats":
            outputs = torch.squeeze(self.att_stats(hidden_states))
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.pretrained_model(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
       
        
        
        hidden_states = outputs[2]

        if self.weight_encoder_layers and self.pool_position == 'after':
          hidden_states = self.weight_encoder_layers(hidden_states)
          hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        elif self.weight_encoder_layers and self.pool_position == 'before':
          hidden_states = [self.merged_strategy(hidden_state, mode=self.pooling_mode) for hidden_state in hidden_states]
          hidden_states = self.weight_encoder_layers(hidden_states)
        else:
          hidden_states = outputs[0]
          hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)



        logits = self.custom_classifier(hidden_states)

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
                loss_fct = CrossEntropyLoss(torch.tensor(self.config.label_weights).cuda())
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                #loss_fct = BCEWithLogitsLoss(torch.tensor(self.config.label_weights))
                #print(labels)
                #print(logits)
                loss_fct = lambda x,y: sum([1-CCCLoss(x[:,i],y[:,i]) for i in range(self.num_labels)])/self.num_labels
                loss = loss_fct(logits, labels)
            if self.config.use_l2_reg:
                loss += self.custom_classifier.l2_regularization_loss()

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

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor, List[List[float]]]]]) -> Dict[str, torch.Tensor]:
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


