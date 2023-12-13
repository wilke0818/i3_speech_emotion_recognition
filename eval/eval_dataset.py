
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
import sys
sys.path.insert(0, './utils')
from model_classes import *
import json
from audiomentations import LoudnessNormalization, Compose, AddGaussianNoise
import csv
import argparse

def run_eval(model_path, save_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    processor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True) #AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = ModelForSpeechClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=config, local_files_only=True).to(device)
    
    
    target_sampling_rate = processor.sampling_rate
    
    
    def test_data_prep(path):
        path = path['path']
#        file_name = path.split('/')[-1]
 #       path = os.path.join('./data/audio4analysis_orig', file_name)
  #      print(path)
   #     print(os.path.abspath(path))
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
    
    
    test_dataset = load_dataset("csv", data_files={"test": os.path.join(save_path, "test.csv")}, delimiter="\t")["test"]
    
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
    output_column = "class_id"
    print(test_dataset) 
    test_dataset = test_dataset.map(test_data_prep)
    
    result = test_dataset.map(predict, batched=True, batch_size=8)
    y_true = []
    y_pred = []
    label_names = [config.id2label[i] for i in range(config.num_labels)]
    
    for x in result:
      if x['class_id'] in label_names:
        y_true.append(config.label2id[x['class_id']])
        y_pred.append(x['predicted'])
    #y_true = [config.label2id[name] for name in result["class_id"] if name in label_names]
    #y_pred = [x for x in result["predicted"] if config.id2label[x]
    

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
    print(genders)
    genders_cm = {}
    accuracies = {'overall': accuracy}
    
    for gender in genders:
      y_true_gender = []
      y_pred_gender = []
      for x in result:
        if x['class_id'] in label_names and x['gender']==gender:
          y_pred_gender.append(x['predicted'])
          y_true_gender.append(config.label2id[x['class_id']])
      if len(y_pred_gender) < .1*len(result): continue
      df_cm_gender = make_cm(emotions, y_true_gender, y_pred_gender, os.path.join(output_path, f'{gender}_confusion_matrix.png'))
      eval_dict_gender = classification_report(y_true_gender, y_pred_gender, target_names=label_names, output_dict=True)
      genders_cm[gender] = df_cm_gender 
      accuracies[gender] = eval_dict_gender['accuracy']
    df_cm = make_cm(emotions, y_true, y_pred, os.path.join(output_path, 'confusion_matrix.png'))

    json_path = os.path.join(output_path, 'accuracies.json')
    with open(json_path, 'w') as f:
      f.write(json.dumps(accuracies))
    return df_cm, accuracy, genders_cm, accuracies

def make_cm(label_names, y_true, y_pred, output_name):
    cm=confusion_matrix(y_true, y_pred)
    print(cm)
    print(len(label_names))
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
    plt.close()
    return df_cm


#An example call to this function, especially to run an evaluation on an already trained model
#run_eval('./jonatasgrosman_wav2vec2_large_xlsr_53_italian/0/', './data/EMOVO/', './outputs/jonatasgrosman_wav2vec2_large_xlsr_53_italian/emovo/0/')
#run_eval('/om2/user/wilke18/model_outputs/jonatasgrosman_wav2vec2_large_xlsr_53_italian_eval_loss_test1/0', './data/EMOVO_norm/', './emovo/')

def main():
  confusion_matrices = {}
  accuracies = {}
  parser=argparse.ArgumentParser()

  parser.add_argument("--eval_path", help="Runs the evaluation code using CSV path given. If directory, uses all CSVs in the path")
  

  args=parser.parse_args()
  evals = get_csv_info(args.eval_path)

  for model in evals.keys():
    eval_base_path = ''
    for dataset in evals[model].keys():
      for (model_path, eval_csv_path, eval_out_path) in evals[model][dataset]:
  
        if dataset not in confusion_matrices.keys():
          confusion_matrices[dataset] = {'overall': []}
          accuracies[dataset] = {'overall': []}

        split_path = os.path.split(eval_out_path)
        temp_eval_base_path = os.path.split(split_path[0])[0] if split_path[1] == '' else split_path[0] #gets rid of the model seed on the path

        split_path = os.path.split(temp_eval_base_path)
        assert split_path[1] == dataset
        temp_eval_base_path = split_path[0] #Gets rid of the dataset

        split_path = os.path.split(temp_eval_base_path)
        assert split_path[1] == model
        temp_eval_base_path = split_path[0] #Gets rid of the model name

        if not eval_base_path:
          eval_base_path = temp_eval_base_path
        else:
          assert eval_base_path == temp_eval_base_path

        if not os.path.exists(eval_out_path):
          os.makedirs(eval_out_path)
        cm, accuracy, gendered_cms, gendered_accuracies = run_eval(model_path, eval_csv_path, eval_out_path)
        #print(gendered_cms)
        #print(gendered_accuracies)
        confusion_matrices[dataset]['overall'].append(cm)
        for gender, gender_cm in gendered_cms.items():
          if gender not in confusion_matrices[dataset].keys():
            confusion_matrices[dataset][gender] = []
            accuracies[dataset][gender] = []
            confusion_matrices[dataset][gender].append(gender_cm)
            accuracies[dataset][gender].append(gendered_accuracies[gender])
        accuracies[dataset]['overall'].append(accuracy)
      
    #For a model architecture/settings in this experiment, create a single, averaged confusion matrix of all runs of it
    for dataset in confusion_matrices.keys():
      for subset in confusion_matrices[dataset].keys():
        df_concat=pd.concat(confusion_matrices[dataset][subset],axis=0)
        data_mean=df_concat.groupby(level=0).mean()
        eval_out_path = os.path.join(eval_base_path, model, dataset, f"{subset}_confusion_matrix.csv")
        data_mean.to_csv(eval_out_path)
        accuracies[dataset][subset] = np.mean(accuracies[dataset][subset])
    json_path = os.path.join(eval_base_path, model, 'accuracies.json')
    with open(json_path, 'w') as f:
      f.write(json.dumps(accuracies))

def get_csv_info(csv_path):
  csvs = []
  if os.path.isdir(csv_path):
    csvs = [os.path.join(csv_path, csv_f) for csv_f in os.listdir(csv_path) if csv_f[-4:] == '.csv']
  else:
    csvs = [csv_path]

  evals = {}

  for csv_f in csvs:
    with open(csv_f, 'r') as csvfile:
      reader = csv.DictReader(csvfile, delimiter='\t')
      for row in reader:
        if row['model_name'] not in evals:
          evals[row['model_name']] = {}
        if row['dataset_name'] not in evals[row['model_name']]:
          evals[row['model_name']][row['dataset_name']] = []
        evals[row['model_name']][row['dataset_name']].append((row['model_path'], row['eval_csv_path'], row['eval_out_path']))
  return evals

if __name__ == "__main__":
    sys.exit(main())
