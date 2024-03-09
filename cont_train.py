
#TODO many redundant imports

import csv
import torch
import numpy as np
import pandas as pd
import torchaudio
import os
import sys, argparse
from collections import OrderedDict
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

from augmentations.augmentations import *
from utils.data_splitter import generate_dataset,load_saved_dataset
from model.inputs.model_input import ModelInputParameters
from experiment_input import ExperimentInputParameters
from utils.model_classes import *
from utils.callbacks import *
from eval.eval_dataset import get_csv_info, run_full_evals 
from eval.graph_analysis import run_graph_analysis

import json
import time

from datetime import datetime
import gc
from sklearn.metrics import classification_report


def run_model(model_params, model_path, model_save_folder, hp_amount_of_data, hp_num_trials, resume_from_prev=False, skip_hp_search=False):
    speaker_independent_scenario = model_params.speaker_independent_scenario
    eval_steps = model_params.eval_steps
    logging_steps = model_params.logging_steps        
    per_device_train_batch_size= model_params.per_device_batch_size
    per_device_eval_batch_size= model_params.per_device_batch_size
    batch_size = model_params.batch_size
    num_proc = 1
    save_steps = model_params.save_steps
    model_output_dir=model_save_folder
    model_name_or_path = model_params.model_path_or_name
    pooling_mode = model_params.pooling_mode
    input_column = model_params.input_column
    output_column = model_params.output_column
   
    print(output_column) 
    #Only run the hyperparameter search if we are not skipping it and if we are not resuming training of a model
    if not resume_from_prev and not skip_hp_search:
        #Generate data for hyperparameter search
        hp_search_train_dataset, hp_search_eval_dataset, hp_search_test_dataset = generate_dataset(
            model_params.seed, model_params.train_test_split, speaker_independent_scenario, True, hp_amount_of_data, model_params.training_data_csv)
    
       #Since test set is not used for hp search, combine eval and train back together and use the original test set for eval: gives the desired train/test split with the amount of data specified

        hp_search_train_dataset = concatenate_datasets([hp_search_train_dataset, hp_search_eval_dataset])
        hp_search_eval_dataset = hp_search_test_dataset

        print(hp_search_train_dataset)
        print(hp_search_eval_dataset)

    #generates train, validation, and test set from the emozionalmente dataset

    #train_dataset, eval_dataset, test_dataset = generate_dataset(model_params.seed, model_params.train_test_split, speaker_independent_scenario, True, data_path=model_params.training_data_path)
    train_dataset, eval_dataset, test_dataset = generate_dataset(model_params.seed, model_params.train_test_split, speaker_independent_scenario, True, 1, model_params.training_data_csv)

    print(train_dataset)  
    if type(output_column) is str:
      label_list = train_dataset.unique(output_column)
      label_list.sort()  # Let's sort it for determinism
      num_labels = len(label_list)
      print(label_list)
    else:
      label_list = output_column
      num_labels = len(label_list)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
#        label2id={label: i for i, label in enumerate(label_list)},
#        id2label={i: label for i, label in enumerate(label_list)},
    )
    setattr(config, 'pooling_mode', pooling_mode)
    
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path,) #AutoProcessor.from_pretrained(model_name_or_path,)
    target_sampling_rate = feature_extractor.sampling_rate
    
    def speech_file_to_array_fn(path):
        #path to each speech file; we resample to the desired sampling rate (16000Hz); assumes audio is monochannel which is true of emozionalmente
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(speech_array)
        speech = speech.squeeze().numpy()
        
    #    CUT = 6 # custom cut at 4 seconds for speeding up the data processing (not necessary)
    #    if len(speech) > target_sampling_rate*CUT:
    #        return speech[:int(target_sampling_rate*CUT)]
        
        return speech
    
    def label_to_id(label, label_list):
        if len(label_list) > 0:
            return label_list.index(label) if label in label_list else -1
    
        return label
    
    def preprocess_function(examples):
        #note examples is in batches from the datasets
        #print(examples)
        #speech_list = [[speech_file_to_array_fn(example[input_column]),example['gender']]  for example in examples]
        speech_list = [speech_file_to_array_fn(example)  for example in examples[input_column]]
        gender_list = [gender for gender in examples['gender']]
        #speech_list = [[speech_list[i],gender_list[i]] for i in range(len(gender_list))]
        if type(output_column) is str:
          target_list = [label_to_id(label, label_list) for label in examples[output_column]]
        else:
          target_list = []
          for label in output_column:
            target_list.append(examples[label])
          target_list = np.array(target_list).T.tolist()
        result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)
    
        return result
    

    #Preprocess all the data by getting the actual audio data and the targeted emotion labels
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
    print(eval_dataset)
    def aug_helper(batch, augmentation):
        #batch is batched data from the training dataset and augmentation is the augmentation of that data that we want to perform
        #augmentations are tuples of type (Compose, bool). The Compose is to apply the augmentation and the boolean is whether the augmentation can take in tensors or not (see torch-audiomentations vs. audiomentations)
        if augmentation[1]:
          #creates a tensor of shape (batch=1,channels=1,data)
          speech_augs_out = [augmentation[0](torch.tensor(data).unsqueeze(0).unsqueeze(0), target_sampling_rate) for data in batch['input_values']]
          speech_augs = [out['samples'].squeeze() for out in speech_augs_out]
        else:
          speech_augs_out = [augmentation[0](np.array(data), target_sampling_rate) for data in batch['input_values']]
          speech_augs = speech_augs_out
        
        return {'input_values': speech_augs}
    
    if len(model_params.augmentations) > 0:
      temp_dataset = None
      #Note this is for union augmentations meaning each augmentation is done on the train dataset and the union of all these resulting datasets is used for training
      #TODO incorporate different augmentation types rather than just Union such as compound augmentations (like applying noise and then pitch shifting the now noisy training data)
      for aug in model_params.augmentations:
          aug_train_dataset = train_dataset.map(
                  aug_helper,
                  fn_kwargs={'augmentation': aug},
                  batch_size=batch_size,
                  batched=True,
                  num_proc=num_proc
                  )
          print(aug)
          #concatenate the augmentated data into a temporary dataset
          if temp_dataset is None:
              temp_dataset = aug_train_dataset
          else:
              temp_dataset = concatenate_datasets([temp_dataset, aug_train_dataset])
    
      print(temp_dataset)
      train_dataset = concatenate_datasets([aug_train_dataset, temp_dataset])
    print(train_dataset) #we should see a size increase if any augmentations were actually used


    data_collator = DataCollatorCTCWithPadding(processor=feature_extractor, padding=True)
    is_regression = model_params.is_regression 
    
    

    def compute_metrics(p: EvalPrediction):
        #print('in compute metrices')
        #print('pred', p.predictions)
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        #preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if is_regression:
            if model_params.eval_metric == "mse":
              return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            elif model_params.eval_metric == "ccc":
              ccc_avg = []
              print(preds.shape)
              for i in range(num_labels):
                p_act = torch.from_numpy(p.label_ids) if isinstance(p.label_ids[:,i],np.ndarray) else p.label_ids[:,i]
                p_pred = p_act = torch.from_numpy(preds[:,i]) if isinstance(preds[:,i], np.ndarray) else preds[:,i]
                ccc_avg.append(CCCLoss(p_pred, p_act))
              return {"ccc": np.array(ccc_avg).mean().item()}
        else:
            if model_params.eval_metric == 'accuracy':
              return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
            elif model_params.eval_metric == 'f1score':
              report = classification_report(p.label_ids, preds, target_names=label_list, output_dict=True)
              return {'f1score': report['weighted avg']['f1-score']}

    def preprocess_logits_for_metrics(logits, labels):
      #preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
      #print('In preprocess_logits')
      #print('logits:', logits)
      preds = torch.squeeze(logits[0]) if is_regression else torch.argmax(logits[0], axis=1)
      #print('preds:', preds)
      return preds, labels

   
    #generate hyperparameter search training arguments
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

        #The number of evaluations to perform early stopping after is calculated to be 10% of the training data
        #TODO: generalize this code further to remove constants
        num_evals = round(len(hp_search_train_dataset)/per_device_train_batch_size/4*15/10)
        quit_after_evals = round(num_evals*.1)
    
    if version.parse(torch.__version__) >= version.parse("1.6"):
        _is_native_amp_available = True
        from torch.cuda.amp import autocast
   
    if not is_regression: 
      label_counts = {k:0 for k in label_list}
      for row in train_dataset:
        label_counts[row['class_id']]+=1
      label_weights = [1/label_counts[label] for label in label_list]
      setattr(config, 'label_weights', label_weights)
    #for label in label_list:
    #  label_weights.append(1/(len(train_dataset.filter(lambda example: example['class_id']==label))))
    setattr(config, 'use_dropout', model_params.use_dropout)
    setattr(config, 'dropout_rate', model_params.dropout_rate)
    setattr(config, 'use_batch_norm', model_params.use_batch_norm)
    setattr(config, 'use_l2_reg', model_params.use_l2_reg)
    setattr(config, 'weight_decay', model_params.weight_decay)
    setattr(config, 'use_weight_encoder_layers', model_params.use_weight_encoder_layers)
    setattr(config, 'pool_position', model_params.pool_position)
    setattr(config, 'use_relu', model_params.use_relu)
    setattr(config, 'is_regression', is_regression)

    def model_init():
        model = ModelForSpeechClassification.from_pretrained(
          model_name_or_path,
          config=config
        ).to('cuda')
        model.freeze_feature_extractor()
        return model
    
    #Uses optuna for hyperparameter searches
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
            tokenizer=feature_extractor,
        )

        print('Quitting after:', quit_after_evals)
        print(trainer.pop_callback(WandbCallback)) #Don't use WandB for tracking
        trainer.add_callback(EarlyStoppingCallback(quit_after_evals)) #Add a callback for early stopping
        trainer.add_callback(MemorySaverCallback) #Add a callback for managing memory during hyperparameter searching; deletes all unessecary saved models

        best_run = trainer.hyperparameter_search(direction="minimize", backend="optuna", hp_space=my_hp_space, compute_objective=my_objective, n_trials=hp_num_trials)
        print(best_run)
    

    steps_per_epoch = (len(train_dataset)//per_device_train_batch_size)//8
    epochs_number = model_params.number_of_training_epochs
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
        #uses the default learning rate
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            evaluation_strategy="steps",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs_number, #Now a model parameter that defaults to 50
            logging_dir=model_params.logs_save_path,
            #fp16=True,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            save_total_limit=model_params.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            seed=int(model_params.seed)
            #eval_delay=steps_per_epoch
        )
    
    #reset (or initialize) the model after hyperparameter search
    model = model_init()
    
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,
    )
 
    num_evals = round(len(train_dataset)/per_device_train_batch_size/4*epochs_number) #Not currently used

    quit_after_evals = 4 #Quit after 50 evals of not seeing the threshold improvement (50 evals = 500 steps; dependent on batch size, and training dataset size) #round(num_evals*0.05) #round(num_evals*.1)
    print('Quitting after:', quit_after_evals)
    early_stopping_threshold = 0.025
    trainer.add_callback(EarlyStoppingCallback(quit_after_evals, early_stopping_threshold))
    trainer.add_callback(EarlyStoppingCallback(3))
    #trainer.add_callback(CustomizedEarlyStoppingCallback(quit_after_evals, early_stopping_threshold, 1.0))
    trainer.add_callback(PrinterCallback()) #Used for logging but not actually necessary now when using tensorflow logs
 
    print(trainer.pop_callback(WandbCallback))
    history = trainer.train(resume_from_checkpoint=resume_from_prev)
    
    #Save the model that was trained to the designated path (differs from the intermediary saves)
    model.save_pretrained(model_path)
    config.save_pretrained(model_path)
    feature_extractor.save_pretrained(model_path)

    model = None
    config = None
    feature_extractor = None
    torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
      torch.cuda.empty_cache()
      gc.collect()



def main():
    experiment = None
    print('Starting experiment')

    parser=argparse.ArgumentParser()

    parser.add_argument("--low_seed", type=int, help="The lowest seed to use. If not set defaults to 0.")
    parser.add_argument("--high_seed", type=int, help="The highest seed to use. If not set defaults to the start seed plus the number of cross validations. Is inclusive.")
    parser.add_argument("--experiment_file", help="The path to the experiment file that controls this experiment run")

    args=parser.parse_args()

    #JSON setup file either defaults to run.json or can be passed as un-named command line argument
    if len(sys.argv) == 2:
      json_file = sys.argv[1]
    elif args.experiment_file:
      json_file = args.experiment_file
    else:
      json_file = 'run.json'

    experiment = ExperimentInputParameters.fromJSON(json_file)
    #with open(json_file) as f:
    #  experiment = json.load(f)

    #assert(experiment.get('experiment_name')) #Don't want to train and then fail cuz we forgot this. will remove when experiment_input class exists
    if not os.path.exists(experiment.model_save_folder):
        os.makedirs(experiment.model_save_folder)

    
    #Uses seeds in the range [start, end). Arguments passed in with high and low seed take precedence over cross_validation
    start = 0
    end = 1

    if args.low_seed:
      assert args.low_seed >= 0
      start = args.low_seed
    
    if args.high_seed:
      assert args.high_seed > start
      end = args.high_seed+1

    if experiment.cross_validation:
      if args.high_seed!=None and not args.low_seed!=None:
        start = end+1 - experiment.cross_validation
      elif args.low_seed!=None and not args.high_seed!=None or (not args.low_seed!=None and not args.high_seed!=None): 
        end = start+experiment.cross_validation
    
    seeds = np.arange(int(start), int(end))

    print(seeds)

    augmentations = []
    
    if experiment.augmentations:
        for (aug_desc, aug) in experiment.augmentations.items():
            aug_name = aug.pop("class")
            module_name = aug.pop("module")
            use_tensors = aug.pop("tensors", True)
            #Each element of augmentation is a tuple of (Compose, bool, str) where this last one is ignored except for debugging
            augmentations.append((create_union_augmentation(module_name,aug_name, aug), use_tensors, aug_desc))
    
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Make sure you see all of the augmentations that were inteded
    print(augmentations)

    evaluations = []

    model_files = os.listdir(experiment.model_input_files) if os.path.isdir(experiment.model_input_files) else []
    model_files = [os.path.join(experiment.model_input_files, model) for model in model_files] if model_files else [experiment.model_input_files]
    print(model_files)

    model_artifacts_top_level = os.path.join(experiment.experiment_artifacts_output_path, experiment.experiment_name)
    logs_main_path = os.path.join(model_artifacts_top_level, 'logs')
    graphs_main_path = os.path.join(model_artifacts_top_level, 'graphs')
    for model in model_files:
        confusion_matrices = {}
        accuracies = {}
        
        
        model_params = ModelInputParameters.fromJSON(model)
        if len(augmentations) > 0:
            model_params.name += "__augmented__union" 

        
        for seed in seeds:
            print(f'Running {model_params.name} seed {seed}')
            torch.cuda.empty_cache()
            gc.collect()

            with torch.no_grad():
              torch.cuda.empty_cache()
              gc.collect()
            #model_params.training_data_path = experiment.get('training_data_path', './data/audio4analysis/')

            model_params.training_data_csv = experiment.training_dataset['data_csv']
            model_params.seed = seed
            model_params.augmentations = augmentations

            model_params.save_steps = experiment.save_steps if experiment.save_steps else experiment.eval_steps
            model_params.logging_steps = experiment.logging_steps
            model_params.eval_steps = experiment.eval_steps
            model_output_path = os.path.join(experiment.model_save_folder,experiment.experiment_name,str(seed),model_params.name)
            print(model_output_path)
            #model_path = os.path.join(model_path, str(seed))
            skip_training = False
            #Make necessary paths for saving models if they don't exist
            #if not os.path.exists(model_path):
            #    os.makedirs(model_path)
            if os.path.exists(model_output_path) and len(os.listdir(model_output_path)) > 0:
                #continue
                skip_training = True

            now = datetime.now()
            time_str = datetime.strftime(now, '%b%d_%H-%M-%S')
            #model_artifacts_top_level = os.path.join(experiment.experiment_artifacts_output_path, experiment.experiment_name)
            #Keeping original logs format for most part to allow for multiple runs of experiment
            #logs_main_path = os.path.join(model_artifacts_top_level, 'logs')
            logs_save_path = os.path.join(logs_main_path, model_params.name, str(seed), 'runs', time_str)
            evals_save_path = os.path.join(model_artifacts_top_level, 'evals')
            #graphs_main_path = os.path.join(model_artifacts_top_level, 'graphs')
            graphs_save_path = os.path.join(graphs_main_path, str(seed), model_params.name)
            checkpoint_output_path = os.path.join(experiment.model_save_folder, experiment.experiment_name, 'checkpoints',str(seed),model_params.name)
            
            os.makedirs(logs_save_path, exist_ok=True)
            os.makedirs(evals_save_path, exist_ok=True)
            os.makedirs(graphs_save_path, exist_ok=True)
            os.makedirs(model_output_path, exist_ok=True)
            os.makedirs(checkpoint_output_path, exist_ok=True)

            model_params.logs_save_path = logs_save_path
            #if not os.path.exists(f'./{model_params.name}/{seed}/'):
            #    os.makedirs(f'./{model_params.name}/{seed}/')
            #model_save_folder = f'./{model_params.name}/{seed}/'

            split_data_path = os.path.split(model_params.training_data_csv)
            training_data_path = os.path.join(split_data_path[0], split_data_path[1][:-4])#os.path.split(model_params.training_data_csv)[0]
            #Default path based off of emozionalmente test set that is generated
            default_eval_csv_path = os.path.join(training_data_path, f'train_test_validation/{seed}/speaker_ind_{model_params.speaker_independent_scenario}_100_{int(100*model_params.train_test_split)}')
            
            resume = model_params.continue_model_training
            skip_hp_search = model_params.skip_hp_search
            model_params.save_total_limit = None if experiment.save_total_limit==-1 else experiment.save_total_limit if experiment.save_total_limit else 1
            model_params.eval_metric = experiment.eval_metric
            print('Setting save_total_limit to', model_params.save_total_limit) 
            #Train the given model
            if not skip_training:
              run_model(model_params, model_output_path, checkpoint_output_path, experiment.hp_amount_of_training_data, 
                        experiment.hp_num_trials, resume, skip_hp_search)
           
            #if experiment.run_grapher:
            #  input_path = os.path.join(logs_main_path, model_params.name)
            #  output_path = os.path.join(graphs_save_path, f'{model_params.name}_seed_{seed}_training_graph.png')
            #  run_graph_analysis(input_path=input_path, output_path=output_path, only_model_folder=True, seed=str(seed), metric=experiment.eval_metric)


            #Changed such that training datasets is no longer in datasets
            eval_out_path = os.path.join(evals_save_path, model_params.name, experiment.training_dataset['name'], str(seed))
            eval_csv_path = default_eval_csv_path
            evaluations.append([model_params.name, experiment.training_dataset['name'], model_output_path, eval_csv_path, eval_out_path])
            #Don't evaluate but create the setup to later run evaluations
            for dataset in experiment.datasets:
              eval_out_path = os.path.join(evals_save_path, model_params.name, dataset['name'], str(seed))
              eval_csv_path = dataset['eval_csv_path']
              evaluations.append([model_params.name, dataset['name'], model_output_path, eval_csv_path, eval_out_path])
    #output_experiment_path = os.path.join(experiment.experiment_artifacts_output_path, experiment.experiment_name)

    if experiment.run_grapher:
        for seed in seeds:
            input_path = logs_main_path
            output_path = os.path.join(graphs_main_path, str(seed), f'{experiment.experiment_name}_seed_{seed}_model_comparison_training_graph.png')
            run_graph_analysis(input_path=input_path, output_path=output_path, only_model_folder=False, seed=str(seed),metric=experiment.eval_metric)

    csv_name = str(int(time.time()))
    with open(os.path.join(evals_save_path, f'{csv_name}.csv'), 'w+') as f:
      csv_file = csv.writer(f, delimiter='\t')
      csv_file.writerow(['model_name', 'dataset_name', 'model_path', 'eval_csv_path', 'eval_out_path'])
      for eval in evaluations:
        csv_file.writerow(eval)

    if experiment.run_eval:
      evals = get_csv_info(os.path.join(evals_save_path, f'{csv_name}.csv'))
      run_full_evals(evals)
if __name__ == "__main__":
    sys.exit(main())
