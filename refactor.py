run_setup = False
if run_setup:
    setup()

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
from transformers import AutoConfig, Wav2Vec2Processor
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

#variables to probably be moved out later

seed = 0
save_path = f'./data/train_test_validation/{seed}/'
speaker_independet_scenario = True
eval_steps = 10
logging_steps = 10
per_device_train_batch_size= 8
per_device_eval_batch_size= 8
batch_size = 8
num_proc = 1
save_steps = eval_steps * 10
model_output_dir="./wav2vec2-xlsr/"
model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
#model_name_or_path = "facebook/wav2vec2-large-xlsr-53-italian"
#model_name_or_path = "facebook/wav2vec2-base-10k-voxpopuli-ft-it"
pooling_mode = "mean"
model_path = "./model/final/"

def my_split(df, data, state=0):
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=state)
    gss.get_n_splits()

    my_real_train_idx = []
    my_real_test_idx = []

    for label in df['emotion'].unique():
        df_data_by_emotion = data[data['emotion'] == label]
        X = df_data_by_emotion
        y = df_data_by_emotion["emotion"]
        groups = df_data_by_emotion['actor']

        for train_idx, test_idx in gss.split(X, y, groups):
            my_real_train_idx = np.concatenate([my_real_train_idx, df_data_by_emotion.iloc[train_idx].index])
            my_real_test_idx = np.concatenate([my_real_test_idx, df_data_by_emotion.iloc[test_idx].index])

    return my_real_train_idx, my_real_test_idx

def generate_and_save_dataset():
    data = []

    for path in tqdm(Path('./data/audio4analysis/').glob("**/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        original_name = str(name).split('____')[-1]
        label = str(name).split('____')[-2]
        actor = str(name).split('____')[-3]
        gender = str(name).split('____')[-4]
    
        try:
            s = torchaudio.load(path)
            data.append({
                "original_name": original_name,
                "name": name,
                "path": path,
                "emotion": label,
                "actor": actor,
                "gender": gender
            })
        except Exception as e:
            # Check if there are some broken files
            print(str(path), e)
    
    df = pd.DataFrame(data)
    df.groupby("emotion").count()[["path"]]

    if speaker_independet_scenario == True:
        males = df[df['gender'] == "male"]
        females = df[df['gender'] == "female"]
    
        real_train_idx, real_test_idx = my_split(df,males,seed)
        males_train_df = df.iloc[real_train_idx.astype(int)]
        males_test_df = df.iloc[real_test_idx.astype(int)]
    
        real_train_idx, real_test_idx = my_split(df,females,seed)
        females_train_df = df.iloc[real_train_idx.astype(int)]
        females_test_df = df.iloc[real_test_idx.astype(int)]
    
        real_train_idx, real_val_idx = my_split(df,males_train_df,seed)
        males_train_df = df.iloc[real_train_idx.astype(int)]
        males_val_df = df.iloc[real_val_idx.astype(int)]
    
        real_train_idx, real_val_idx = my_split(df,females_train_df,seed)
        females_train_df = df.iloc[real_train_idx.astype(int)]
        females_val_df = df.iloc[real_val_idx.astype(int)]
    
    #     print(males_train_df)
    #     print(females_train_df)
    
    
        train_df = pd.concat([males_train_df,females_train_df])
    
    #     print(train_df)
        test_df = pd.concat([males_test_df,females_test_df])
        val_df = pd.concat([males_val_df,females_val_df])
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["emotion"])
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=seed, stratify=train_df["emotion"])


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
    val_df.to_csv(f"{save_path}/val.csv", sep="\t", encoding="utf-8", index=False)

def load_saved_dataset():
    data_files = {
        "train": save_path + "/train.csv",
        "validation": save_path + "/val.csv",
        "test": save_path + "/test.csv",
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    return train_dataset, eval_dataset, test_dataset

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    CUT = 4 # custom cut at 4 seconds for speeding up the data processing (not necessary)
    if len(speech) > 16000*CUT:
        return speech[:int(16000*CUT)]
    return speech

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    result = processor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relu=nn.ReLU()
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.relu(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

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
        outputs = self.wav2vec2(
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
    processor: Wav2Vec2Processor
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

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


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
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


