import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import lightning as L
from torch.utils.data import random_split, DataLoader
from torch.nn import functional as F
from lightning import Trainer
import datasets as d
import csv
import numpy as np
from sklearn.metrics import classification_report
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import argparse

import re

import json

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

class CustomAudioEmbeddingsDataset(Dataset):
  def __init__(self, metadata_csv, embeddings_dir):
    super().__init__()
    with open(metadata_csv) as f:
      dialect = csv.Sniffer().sniff(f.read(), delimiters=';,\t')
      f.seek(0)
      reader = csv.DictReader(f, dialect=dialect)
    self.metadata = d.Dataset.from_pandas(pd.read_csv(metadata_csv, dialect=dialect))
    self.embeddings_dir = embeddings_dir
    label_list = self.metadata.unique('class_id')
    label_list.sort()  # Let's sort it for determinism
    self.num_labels = len(label_list)
    self.label2id={label: i for i, label in enumerate(label_list)}
    self.id2label={i: label for i, label in enumerate(label_list)}
    

  def __len__(self):
    return len(self.metadata['class_id'])

  def __getitem__(self, idx):
    audio_path = os.path.join(self.embeddings_dir, os.path.split(self.metadata['path'][idx])[1])
    audio_path = audio_path.replace(".wav", ".npy")
    audio_embeddings = np.squeeze(np.load(audio_path))
    labels = self.label2id[self.metadata['class_id'][idx]]
    return audio_embeddings, labels

class CustomAudioEmbeddingsDataloader(L.LightningDataModule):
  def __init__(self, embeddings_dir, metadata_csv, batch_size=32):
    super().__init__()
    self.embeddings_dir = embeddings_dir
    self.metadata_csv = metadata_csv
    self.batch_size = batch_size

  def setup(self, stage):
    
    full_dataset = CustomAudioEmbeddingsDataset(self.metadata_csv, self.embeddings_dir)
    self.train_set, self.val_set, self.test_set = random_split(full_dataset, [.8,.1,.1], generator=torch.Generator().manual_seed(42))
    self.input_size = len(full_dataset[0][0])
    self.output_size = full_dataset.num_labels
  
  def train_dataloader(self):
    return DataLoader(self.train_set, batch_size=self.batch_size)

  def val_dataloader(self):
    return DataLoader(self.val_set, batch_size=self.batch_size)

  def test_dataloader(self):
    return DataLoader(self.test_set, batch_size=self.batch_size)

class AudioEmbeddingsClassifier(L.LightningModule):
  def __init__(self, embedding_size, num_classes):
    super().__init__()
    self.l1 = torch.nn.Linear(embedding_size, num_classes)

  def forward(self, x):
    return self.l1(x.view(x.size(0), -1))

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x,y = batch
    y_hat = self(x)
    #print('x', x)
    #print('y', y)
    #print('y_hat', y_hat)
    loss = F.cross_entropy(y_hat, y)
    self.log('val_loss', loss)
    return loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    report = classification_report(y, pred_ids, output_dict=True)
    loss = report['weighted avg']['f1-score']
    self.log('f1score', loss)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.02)

def run_lin_classifier(embeddings_dir, metadata_csv):
  
  audio_data_module = CustomAudioEmbeddingsDataloader(embeddings_dir, metadata_csv)
  audio_data_module.setup(stage='fit')
  model = AudioEmbeddingsClassifier(embedding_size=audio_data_module.input_size, num_classes=audio_data_module.output_size) 

  trainer = Trainer(accelerator="gpu", callbacks=[EarlyStopping(monitor="val_loss", mode="min", min_delta=0.025, patience=2)])
  trainer.fit(model, audio_data_module.train_dataloader(), audio_data_module.val_dataloader())
  return trainer.test(model, audio_data_module.test_dataloader())

def main():
  parser=argparse.ArgumentParser()

  parser.add_argument("--experiment_path", help="Path to experiment folder with the embeddings")
  parser.add_argument("--metadata_csv", help="Path to the CSV containing labels and files")
  parser.add_argument("--output_path", help="Path to output the resulting linear classifier scores. Should be a JSON file.")  

  args=parser.parse_args() 
  seeds = os.listdir(args.experiment_path)
  lin_class_results = {}
  for seed in seeds:
    temp_path = os.path.join(args.experiment_path, seed)
    lin_class_results[seed] = {}
    model_types = os.listdir(temp_path)
    for model in model_types:
      temp_path = os.path.join(args.experiment_path, seed, model)
      lin_class_results[seed][model] = {}
      checkpoints = os.listdir(temp_path)
      for checkpoint in checkpoints:
        temp_path = os.path.join(args.experiment_path, seed, model, checkpoint)
        lin_class_results[seed][model][checkpoint] = {}
        layers = os.listdir(temp_path)
        for layer in layers:
          temp_path = os.path.join(args.experiment_path, seed, model, checkpoint, layer)
          print(temp_path)
          lin_class_results[seed][model][checkpoint][layer] = run_lin_classifier(temp_path, args.metadata_csv)

  print(lin_class_results)
  with open(args.output_path, 'w') as f:
    json.dump(lin_class_results, f)



if __name__ == "__main__":
    main()
