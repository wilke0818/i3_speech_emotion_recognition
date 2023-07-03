#!/usr/bin/env python
# coding: utf-8

# # Speech Emotion Recognition in Italian using Wav2Vec 2.0

# In this notebook, we show how we used wav2vec 2.0 to recognize the emotional aspects of speech in Italian.
# In the study, we used the acted emotional speech corpus Emozionalmente for training, testing and validating the model ([the dataset can be easily downloaded from this link](https://doi.org/10.5281/zenodo.6569824)). Please cite "Emozionalmente: a crowdsourced Italian speech emotional corpus" by Fabio Catania if you use this dataset. Our work was inspired a lot by this amazing [previous work](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb#scrollTo=nEKDAy2bCGFM) on the Greek language (**thank you to the authors!!**).

# We start installing and importing some software packages and defining some enviroment values.

# In[1]:


#get_ipython().run_line_magic('pip', 'install torch')
#get_ipython().run_line_magic('pip', 'install torchaudio')
#get_ipython().run_line_magic('pip', 'install tensorflow')
#get_ipython().run_line_magic('pip', 'install sklearn')
#get_ipython().run_line_magic('pip', 'install scikit-learn')
#get_ipython().run_line_magic('pip', 'install seaborn')
#get_ipython().run_line_magic('pip', 'install datasets')
#get_ipython().run_line_magic('pip', 'install optuna')
#get_ipython().run_line_magic('pip', 'install pytorch-lightning')
#get_ipython().run_line_magic('pip', 'install tqdm')
#get_ipython().run_line_magic('pip', 'install transformers')
#get_ipython().run_line_magic('pip', 'install requests')
#get_ipython().run_line_magic('pip', 'install wandb')
#get_ipython().run_line_magic('pip', 'install datasets')
#get_ipython().run_line_magic('pip', 'install soundfile')
#get_ipython().run_line_magic('pip', 'install accelerate')


# In[2]:


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

#if is_apex_available():
  # from apex import amp


# We also set some parameters we use to setup our model and study.

# In[3]:


# tuning some parameters
speaker_independet_scenario = True
eval_steps = 10
logging_steps = 10
per_device_train_batch_size= 16
per_device_eval_batch_size= 16
batch_size = 8
num_proc = 1
save_steps = eval_steps * 10
model_output_dir="./wav2vec2-xlsr/"
model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
#model_name_or_path = "facebook/wav2vec2-large-xlsr-53-italian"
#model_name_or_path = "facebook/wav2vec2-base-10k-voxpopuli-ft-it"
pooling_mode = "mean"
model_path = "./model/final/"


# We download the Emozionalmente corpus and prepare it for analysis.
# We uploaded the dataset on the Dropbox to make this tutorial accessible.

# In[4]:

# In[4]:


users_df = pd.read_csv("data/metadata/users.csv")
users_df


# In[5]:


samples_df = pd.read_csv("data/metadata/samples.csv")
samples_df


# We pre-process the data to prepare it for analysis.

# In[5]:


get_ipython().system('mkdir -p data/audio4analysis')

samples_df['actor_gender'] = None
samples_df['actor_age'] = None
samples_df['new_file_name'] = None

for index, sample in samples_df.iterrows():
  actor = sample['actor']
  file_name = sample['file_name']
  emotion_expressed = sample['emotion_expressed']
  age = users_df[users_df['username'] == actor]['age'].values[0]
  gender = users_df[users_df['username'] == actor]['gender'].values[0]
  new_file_name = gender + '____' + actor + '____' + emotion_expressed + '____' + file_name
  samples_df.iloc[index, samples_df.columns.get_loc('actor_age')] = age
  samples_df.iloc[index, samples_df.columns.get_loc('actor_gender')] = gender
  samples_df.iloc[index, samples_df.columns.get_loc('new_file_name')] = new_file_name

# In[6]:


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


# For training purposes, we split data into train, test, and validation sets, keeping a proportional balance among emotions and actors' genders. Also, if `speaker_independet_scenario == True`, actors in the train set are not present in the test and validation sets (to prevent overfitting).

# In[7]:


def my_split(data):
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
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


# In[8]:


if speaker_independet_scenario == True:
    males = df[df['gender'] == "male"]
    females = df[df['gender'] == "female"]

    real_train_idx, real_test_idx = my_split(males)
    males_train_df = df.iloc[real_train_idx.astype(int)]
    males_test_df = df.iloc[real_test_idx.astype(int)]

    real_train_idx, real_test_idx = my_split(females)
    females_train_df = df.iloc[real_train_idx.astype(int)]
    females_test_df = df.iloc[real_test_idx.astype(int)]

    real_train_idx, real_val_idx = my_split(males_train_df)
    males_train_df = df.iloc[real_train_idx.astype(int)]
    males_val_df = df.iloc[real_val_idx.astype(int)]

    real_train_idx, real_val_idx = my_split(females_train_df)
    females_train_df = df.iloc[real_train_idx.astype(int)]
    females_val_df = df.iloc[real_val_idx.astype(int)]
    
#     print(males_train_df)
#     print(females_train_df)
    

    train_df = pd.concat([males_train_df,females_train_df])
    
#     print(train_df)
    test_df = pd.concat([males_test_df,females_test_df])
    val_df = pd.concat([males_val_df,females_val_df])
else:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, stratify=df["emotion"])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0, stratify=train_df["emotion"])


# In[9]:


save_path = "./data/train_test_validation/"

if not os.path.exists(save_path):
    os.makedirs(save_path)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
val_df.to_csv(f"{save_path}/val.csv", sep="\t", encoding="utf-8", index=False)


# In[10]:


data_files = {
    "train": save_path + "/train.csv",
    "validation": save_path + "/val.csv",
    "test": save_path + "/test.csv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]


# In order to preprocess the audio into our classification model, we set up the relevant Wav2Vec2 assets regarding the [pretrained Italian model](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-italian). To handle the context representations in any audio length we use a merge strategy plan (pooling mode) to concatenate that 3D representations into 2D representations (`mean`).

# In[11]:


input_column = "path"
output_column = "emotion"


# In[12]:


label_list = train_dataset.unique(output_column)
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)

# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
)
setattr(config, 'pooling_mode', pooling_mode)


# In[13]:


processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
target_sampling_rate = processor.feature_extractor.sampling_rate


# We use the wav2vec pretrained model to extract features from the audio in context representation tensors.

# In[14]:


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


# In[15]:


train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=batch_size,
    batched=True,
    num_proc=num_proc
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=batch_size,
    batched=True,
    num_proc=num_proc
)


# We build our classification model.

# In[16]:


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# In[17]:


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


# We set up the training pipeline.

# In[18]:


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


# In[19]:


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
is_regression = False


# In[20]:


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


# In[21]:


training_args = TrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="steps",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=2,
    # fp16=True,
    save_steps=save_steps,
    eval_steps=eval_steps,
    logging_steps=logging_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)


# In[22]:


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


# In[23]:


def model_init():
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
      model_name_or_path,
      config=config
    ).to('cuda')
    model.freeze_feature_extractor()
    return model


# We do hyperparameters tuning.

# In[ ]:


def my_hp_space(trial):
    return {
      "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
      "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10),
    }

def my_objective(metrics):
    return metrics["eval_loss"]

trainer = CTCTrainer(
    model_init=model_init,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)
best_run = trainer.hyperparameter_search(direction="minimize", hp_space=my_hp_space, compute_objective=my_objective, n_trials=1)
print(best_run)


# We train our model for emotion classification.

# In[ ]:


training_args = TrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="steps",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=2,
    num_train_epochs= best_run.hyperparameters['num_train_epochs'],
    # fp16=True,
    save_steps=save_steps,
    eval_steps=eval_steps,
    logging_steps=logging_steps,
    learning_rate= best_run.hyperparameters['learning_rate'],
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)


# In[ ]:


model = model_init()


# In[ ]:


trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)


# In[ ]:


history = trainer.train()


# We save our trained emotional classifier.

# In[ ]:


model.save_pretrained(model_path)
config.save_pretrained(model_path)
processor.save_pretrained(model_path)


# We evaluate the performance of our model for speech emotion recognition.
# 
# *If you want to use our best pretrained model (81% of accuracy on the validation set), please uncomment the following line.*

# In[ ]:


'''
!mkdir -p /content/model/pretrained/
!wget -O pretrained-model.zip https://www.dropbox.com/s/abiyfxo2c3a9jcu/pretrained-model.zip
!unzip pretrained-model.zip
!mv "/content/pretrained-model" /content/model/
!rm /content/pretrained-model.zip

model_path = "/content/model/pretrained-model/"
'''


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# In[ ]:


config = AutoConfig.from_pretrained(model_path, local_files_only=True)
processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_path, local_files_only=True).to(device)


# In[ ]:


test_dataset = load_dataset("csv", data_files={"test": save_path +"test.csv"}, delimiter="\t")["test"]
test_dataset


# In[ ]:


def predict(batch):
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch


# In[ ]:


test_dataset = test_dataset.map(speech_file_to_array_fn)


# In[ ]:


result = test_dataset.map(predict, batched=True, batch_size=8)


# In[ ]:


label_names = [config.id2label[i] for i in range(config.num_labels)]
y_true = [config.label2id[name] for name in result["emotion"]]
y_pred = result["predicted"]


# In[ ]:


classification_report(y_true, y_pred, target_names=label_names)


# In[ ]:


emotions = label_names
cm=confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(cm.astype('float') / cm.sum(axis=1), index = [i for i in emotions],
                  columns = [i for i in emotions])
l = 16
fig, ax = plt.subplots(figsize=(l,l/2))
sn.heatmap(df_cm, annot=True, cmap="coolwarm")
ax.set(xlabel='Recognized emotion', ylabel='Expressed emotion')
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)

# We change the fontsize of minor ticks label
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)

plt.show()


# That's all, folks!

# In[ ]:




