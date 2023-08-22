
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

from sklearn.model_selection import GroupShuffleSplit
from datasets import load_dataset, load_metric
from transformers import AutoConfig, AutoModel, AutoProcessor, Wav2Vec2Processor, PreTrainedModel, AutoFeatureExtractor
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

import optuna
from optuna.trial import TrialState
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from transformers import (
    Trainer,
    is_apex_available,
    )
from sklearn.metrics import classification_report
from utils import *
import json
from audiomentations import LoudnessNormalization, Compose, AddGaussianNoise


def run_eval(model_path, save_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    processor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True) #AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = ModelForSpeechClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=config, local_files_only=True).to(device)
    
    
    target_sampling_rate = processor.sampling_rate
    
    
    def test_data_prep(path):
        path = path['path']
        file_name = path.split('/')[-1]
        path = os.path.join('./data/audio4analysis_orig', file_name)
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        augment = Compose([
          LoudnessNormalization(min_lufs=-23, max_lufs=-23, p=1)
        ])
        
        if len(speech.shape) > 1 and speech.shape[0]>1: #if not monochannel, average the channels together
            speech = np.mean(speech, 0)

        speech = augment(speech, target_sampling_rate)    
        CUT = 4 # custom cut at 4 seconds for speeding up the data processing (not necessary)
        #if len(speech) > 16000*CUT:
            #print(speech)
        #    return {'speech': speech[0:int(16000*CUT)]}
        return {'speech': speech}
    
    
    test_dataset = load_dataset("csv", data_files={"test": save_path +"test.csv"}, delimiter="\t")["test"]
    
    def predict(batch):
        features = processor(batch['speech'], sampling_rate=processor.sampling_rate, return_tensors="pt", padding=True)
        
        input_values = features.input_values.to(device)
        attention_mask = features.attention_mask.to(device)
        #print(input_values)
        with torch.no_grad():
            model_result = model(input_values, attention_mask=attention_mask)
            logits = model_result.logits
    
        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        batch["predicted"] = pred_ids
        return batch
    
    input_column = "path"
    output_column = "emotion"
    
    test_dataset = test_dataset.map(test_data_prep)
    
    result = test_dataset.map(predict, batched=True, batch_size=8)
    
    label_names = [config.id2label[i] for i in range(config.num_labels)]
    y_true = [config.label2id[name] for name in result["emotion"]]
    y_pred = result["predicted"]


    accuracy = 0
    with open(os.path.join(output_path, 'classification_report.json'), 'w') as f:
        eval_dict = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
        eval_dict['y_pred'] = y_pred
        eval_dict['y_true'] = y_true
        eval_json = json.dumps(eval_dict)
        accuracy = eval_dict['accuracy']
        f.write(eval_json)
    
    emotions = label_names
    genders = set([x['gender'] for x in result])
    genders_cm = {}
    for gender in genders:
      y_true_gender = [config.label2id[x['emotion']] for x in result if x['gender']==gender]
      y_pred_gender = [x['predicted'] for x in result if x['gender']==gender]
      df_cm_gender = make_cm(emotions, y_true_gender, y_pred_gender, os.path.join(output_path, f'{gender}_confusion_matrix.png'))
      genders_cm[gender] = df_cm_gender 
    
    df_cm = make_cm(emotions, y_true, y_pred, os.path.join(output_path, 'confusion_matrix.png'))
    return df_cm, accuracy, df_cm_gender

def make_cm(label_names, y_true, y_pred, output_name):
    cm=confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cm.astype('float') / cm.sum(axis=1, keepdims=True), index = [i for i in label_names],
                      columns = [i for i in label_names])
    l = 16
    fig, ax = plt.subplots(figsize=(l,l/2))
    sn.heatmap(df_cm, annot=True, cmap="coolwarm")
    ax.set(xlabel='Recognized emotion', ylabel='Expressed emotion')
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)

    # We change the fontsize of minor ticks label
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    fig.savefig(output_name)
    return df_cm


#An example call to this function, especially to run an evaluation on an already trained model
#run_eval('./anon_synt/', './data/audio4analysis_norm_synt/train_test_validation/0/speaker_ind_True_100_80/', './outputs/anon_synt/emozionalmente_copy/')

