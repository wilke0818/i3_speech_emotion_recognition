from eval_dataset import run_eval

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
from datasets import load_dataset, load_metric, concatenate_datasets
from transformers import AutoConfig, AutoProcessor, AutoModel, Wav2Vec2Processor, AutoFeatureExtractor
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
from transformers.integrations import WandbCallback
from transformers import EarlyStoppingCallback

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

from augmentations import *
from data import generate_dataset,load_saved_dataset
from model_input import ModelInputParameters
from utils import *
import json
import sys


def run_model(model_params, model_path, output_path, hp_amount_of_data, hp_num_trials, resume_from_prev=False, skip_hp_search=False):
    speaker_independent_scenario = model_params.speaker_independent_scenario
    eval_steps = model_params.eval_steps
    logging_steps = model_params.logging_steps
    
    
    
    per_device_train_batch_size= model_params.per_device_batch_size
    per_device_eval_batch_size= model_params.per_device_batch_size
    batch_size = model_params.batch_size
    num_proc = 1
    save_steps = eval_steps * 1
    model_output_dir=output_path
    model_name_or_path = model_params.model_path_or_name
    pooling_mode = model_params.pooling_mode
    
    
    if not resume_from_prev and not skip_hp_search:
        hp_search_train_dataset, hp_search_eval_dataset, hp_search_test_dataset = generate_dataset(
            model_params.seed, model_params.train_test_split, speaker_independent_scenario, True, hp_amount_of_data)
    
    #Since test set is not used for hp search, combine eval and train back together and use the original test set for eval: gives the desired train/test split with the amount of data specified

        hp_search_train_dataset = concatenate_datasets([hp_search_train_dataset, hp_search_eval_dataset])
        hp_search_eval_dataset = hp_search_test_dataset

        print(hp_search_train_dataset)
        print(hp_search_eval_dataset)

    #model_params.train_test_split = 0.2
    train_dataset, eval_dataset, test_dataset = generate_dataset(model_params.seed, model_params.train_test_split, speaker_independent_scenario, True)
     
    input_column = model_params.input_column
    output_column = model_params.output_column
    
    
    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        #finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'pooling_mode', pooling_mode)
    
    
    processor = AutoFeatureExtractor.from_pretrained(model_name_or_path,) #AutoProcessor.from_pretrained(model_name_or_path,)
    target_sampling_rate = processor.sampling_rate #processor.feature_extractor.sampling_rate
    
    def speech_file_to_array_fn(path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(speech_array)
        speech = speech.squeeze().numpy()
        '''
        CUT = 4 # custom cut at 4 seconds for speeding up the data processing (not necessary)
        if len(speech) > 16000*CUT:
            return speech[:int(16000*CUT)]
        '''
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
    
    if not resume_from_prev and not skip_hp_search:
        hp_search_train_dataset = hp_search_train_dataset.map(
            preprocess_function,
            batch_size=batch_size,
            batched=True,
            num_proc=num_proc
        )
        hp_search_eval_dataset = hp_search_eval_dataset.map(
            preprocess_function,
            batch_size=batch_size,
            batched=True,
            num_proc=num_proc
        )
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

    print(train_dataset)
    def aug_helper(batch, augmentation):
        if augmentation[1]:
          speech_augs_out = [augmentation[0](torch.tensor(data).unsqueeze(0).unsqueeze(0), target_sampling_rate) for data in batch['input_values']]
          speech_augs = [out['samples'].squeeze() for out in speech_augs_out]
        else:
          speech_augs_out = [augmentation[0](np.array(data), target_sampling_rate) for data in batch['input_values']]
          speech_augs = speech_augs_out
        
        return {'input_values': speech_augs}
    
    if len(model_params.augmentations) > 0:
      temp_dataset = None
      for aug in model_params.augmentations:
          aug_train_dataset = train_dataset.map(
                  aug_helper,
                  fn_kwargs={'augmentation': aug},
                  batch_size=batch_size,
                  batched=True,
                  num_proc=num_proc
                  )
           #print(aug_train_dataset[0]['input_values'])
           #raise("stop it")
          print(aug)
          if temp_dataset is None:
              temp_dataset = aug_train_dataset
          else:
              temp_dataset = concatenate_datasets([temp_dataset, aug_train_dataset])
    
      print(temp_dataset)
      train_dataset = concatenate_datasets([aug_train_dataset, temp_dataset])
    print(train_dataset)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    is_regression = model_params.is_regression
    
    
    # In[20]:
    
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    
        if is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
   
    if not resume_from_prev and not skip_hp_search:
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=4,
            # fp16=True,
            save_strategy='steps',
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            num_train_epochs=15,
            metric_for_best_model="eval_loss",
        )

        num_evals = round(len(hp_search_train_dataset)/per_device_train_batch_size/4*15/10)
        quit_after_evals = round(num_evals*.1)
    
    if version.parse(torch.__version__) >= version.parse("1.6"):
        _is_native_amp_available = True
        from torch.cuda.amp import autocast
    
    def model_init():
        model = ModelForSpeechClassification.from_pretrained(
          model_name_or_path,
          config=config
        ).to('cuda')
        model.freeze_feature_extractor()
        return model
    
    def my_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
        }
        
    def my_objective(metrics):
        return metrics["eval_loss"]
        
    if not resume_from_prev and not skip_hp_search: 
        trainer = CTCTrainer(
            model_init=model_init,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=hp_search_train_dataset,
            eval_dataset=hp_search_eval_dataset,
            tokenizer=processor.feature_extractor,
        )

        print('Quitting after:', quit_after_evals)
        print(trainer.pop_callback(WandbCallback))
        trainer.add_callback(EarlyStoppingCallback(quit_after_evals))
        trainer.add_callback(MemorySaverCallback)

        best_run = trainer.hyperparameter_search(direction="minimize", backend="optuna", hp_space=my_hp_space, compute_objective=my_objective, n_trials=hp_num_trials)
        print(best_run)
    
    epochs_number = 50
    if not resume_from_prev and not skip_hp_search:
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            evaluation_strategy="steps",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs_number,
            # fp16=True,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            learning_rate= best_run.hyperparameters['learning_rate'],
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
    else:
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            evaluation_strategy="steps",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs_number, # TODO: CHANGE 50
            #fp16=True,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy" #"eval_loss",
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
        tokenizer=processor #processor.feature_extractor,
    )
    
    
    # In[ ]:
 
    num_evals = round(len(train_dataset)/per_device_train_batch_size/4*epochs_number)
    quit_after_evals = 50 #round(num_evals*0.05) #round(num_evals*.1)
    print('Quitting after:', quit_after_evals)
    early_stopping_threshold = 0.01
    trainer.add_callback(EarlyStoppingCallback(quit_after_evals, early_stopping_threshold))
    trainer.add_callback(PrinterCallback())
 
    print(trainer.pop_callback(WandbCallback))

   

    history = trainer.train(resume_from_checkpoint=resume_from_prev)
    
    
    model.save_pretrained(model_path)
    config.save_pretrained(model_path)
    processor.save_pretrained(model_path)

def main():
    models = None

    print(sys.argv)
    if len(sys.argv) == 1:
      json_file = 'run.json'
    elif len(sys.argv) == 2:
      json_file = sys.argv[1]
    else:
      json_file = sys.argv[1]
      print("Wasn't expecting more than one argument. Using the first as the json file")

    with open(json_file) as f:
      print(json_file)
      models = json.load(f)

    if not os.path.exists(models['output_path']):
        os.makedirs(models['output_path'])

    if models.get('cross_validation', None) and models['cross_validation']>1:
        rng = np.random.default_rng()
        seeds = rng.permutation(1000)[0:models['cross_validation']]
    else:
        seeds = np.arange(1)

    augmentations = []
    
    if models.get('augmentations', None):
        for (aug_desc, aug) in models['augmentations'].items():
            aug_name = aug.pop("class")
            module_name = aug.pop("module")
            use_tensors = aug.pop("tensors", True)
            augmentations.append((create_union_augmentation(module_name,aug_name, aug), use_tensors, aug_desc))
    
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_samples = torch.ones(size=(8, 1, 32000), dtype=torch.float32, device=torch_device) +1.0

    print(audio_samples)
#    for aug in augmentations:
#        print(aug[2])        
        # Make an example tensor with white noise.
        # This tensor represents 8 audio snippets with 2 channels (stereo) and 2 s of 16 kHz audio.
#        if aug[1]:
#          print(aug[0](audio_samples, 16000))
#        else:
#          test = audio_samples.numpy()
#          print(test.shape)
#          print(aug[0](audio_samples.numpy()[0], 16000))
    print(augmentations)
#    return
    resume = False
    skip_hp_search = True

    for model in os.listdir(models['path_to_model_files']):
        confusion_matrices = {}
        accuracies = {}
        
        
        model_params = ModelInputParameters.fromJSON(os.path.join(models['path_to_model_files'],model))
        if models.get('augmentations', None):
            model_params.name += "__augmented__union" 


        seeds = np.arange(models['cross_validation'])
        for seed in seeds:
            model_params.seed = seed
            model_params.augmentations = augmentations
            model_path = os.path.join(models['output_path'],model_params.name)
            model_path = os.path.join(model_path, str(seed))

            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if not os.path.exists(f'./{model_params.name}/{seed}/'):
                os.makedirs(f'./{model_params.name}/{seed}/')
            output_path = f'./{model_params.name}/{seed}/'

            default_eval_csv_path = f'./data/train_test_validation/{seed}/speaker_ind_{model_params.speaker_independent_scenario}_100_{int(100*model_params.train_test_split)}/'
            run_model(model_params, model_path, output_path, models['hp_amount_of_training_data'], 
                      models['hp_num_trials'], resume, skip_hp_search)
            
            for dataset in models['datasets']:
                if dataset['name'] not in confusion_matrices.keys():
                    confusion_matrices[dataset['name']] = []
                    accuracies[dataset['name']] = []

                eval_out_path = os.path.join('./outputs/', model_params.name, dataset['name'], str(seed))
                if not os.path.exists(eval_out_path):
                    os.makedirs(eval_out_path)
                eval_csv_path = dataset.get('eval_csv_path', default_eval_csv_path)
                cm, accuracy = run_eval(model_path, eval_csv_path, eval_out_path)

                confusion_matrices[dataset['name']].append(cm)
                accuracies[dataset['name']].append(accuracy)
        for dataset in confusion_matrices.keys():
            df_concat=pd.concat(confusion_matrices[dataset],axis=0)
            data_mean=df_concat.groupby(level=0).mean()
            eval_out_path = os.path.join('./outputs/', model_params.name, dataset, 'confusion_matrix.csv')
            data_mean.to_csv(eval_out_path)
            accuracies[dataset] = np.mean(accuracies[dataset])
        json_path = os.path.join('./outputs/', model_params.name, 'accuracies.json')
        with open(json_path, 'w') as f:
            f.write(json.dumps(accuracies))


if __name__ == "__main__":
    sys.exit(main())
