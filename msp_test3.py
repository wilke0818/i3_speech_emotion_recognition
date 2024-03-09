import os
import json
import argparse
import torchaudio
import csv
import pandas as pd
import torch

from utils.model_classes import *
from transformers import AutoFeatureExtractor
parser=argparse.ArgumentParser()

problem = 'continuous'

parser.add_argument("--output_path", help="Where to put/call the metadata csv")
parser.add_argument("--labels_file", help="Path to the labels csv", default="/om2/scratch/Tue/fabiocat/MSP/Partitions.txt")
parser.add_argument("--input_path", help="Path to .wav files referenced in labels_csv")
parser.add_argument("--split_by_set", help="Splits into the sets defined by MSP Podcast", action="store_true")
parser.add_argument("--model_path")

args=parser.parse_args()

test3 = []
with open(args.labels_file) as f:
  for row in f.readlines():
    if 'Test3' not in row:
      continue
    test3.append(os.path.join(args.input_path,row.replace('Test3; ', '').strip()))

from audiomentations import LoudnessNormalization, Compose, AddGaussianNoise
import csv
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
processor = AutoFeatureExtractor.from_pretrained(args.model_path, local_files_only=True) #AutoProcessor.from_pretrained(model_path, local_files_only=True)
model = ModelForSpeechClassification.from_pretrained(pretrained_model_name_or_path=args.model_path, config=config, local_files_only=True).to(device)
    
    
target_sampling_rate = processor.sampling_rate
    
    
def test_data_prep(path):
  speech_array, sampling_rate = torchaudio.load(path)
  resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
  speech = resampler(speech_array).squeeze().numpy()
  augment = Compose([
    LoudnessNormalization(min_lufs=-23, max_lufs=-23, p=1)
  ])
        
  if len(speech.shape) > 1 and speech.shape[0]>1: #if not monochannel, average the channels together
    speech = np.mean(speech, 0)

  speech = augment(speech, target_sampling_rate)    
  return speech
    
    
def predict(speech):
  features = processor(speech, sampling_rate=processor.sampling_rate, return_tensors="pt", padding=True)
        
  input_values = features.input_values.to(device)
  attention_mask = features.attention_mask.to(device)
        #print(input_values)
  with torch.no_grad():
    model_result = model(input_values, attention_mask=attention_mask)
    logits = model_result.logits
    if problem != 'continuous':
      pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy().item()
    else:
      pred_ids = logits.detach().cpu().numpy()
      
      for i in range(len(pred_ids)):
        if pred_ids[i] < 1:
          pred_ids[i] = 1
        elif pred_ids[i] > 7:
          pred_ids[i] = 7
    
    return pred_ids
    
input_column = "path"
output_column = "class_id"
test_dataset = [test_data_prep(test3_path) for test3_path in test3]

    
result = [predict(test_data) for test_data in test_dataset]
print(result[0])

y_pred = []
print(config)
if problem != 'continuous':
  label_names = [config.id2label[i] for i in range(config.num_labels)]
    
  for x in result:
    print(x, type(x))
      #y_true.append(config.label2id[x['class_id']])
    y_pred.append(config.id2label[x])
 
  print(test3)

  with open(args.output_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['FileName', 'EmoClass'])
    for i in range(len(test3)):
      writer.writerow([os.path.split(test3[i])[1], y_pred[i]])
else:
  with open(args.output_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['FileName', 'EmoAct', 'EmoVal', 'EmoDom'])
    for i in range(len(test3)):
      writer.writerow([os.path.split(test3[i])[1], result[i][0], result[i][1], result[i][2]])
