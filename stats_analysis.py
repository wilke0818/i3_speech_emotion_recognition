import numpy as np
import os
import json

results = {}

#TODO make modular
output_path = './outputs'
models = os.listdir(output_path)



for model in models:
  model_path = os.path.join(output_path, model)
  if not os.path.isdir(model_path):
    continue
  #Right now just prints model data as mean +/- standard deviation for each dataset
  print(model)
  results[model] = {}
  for dataset in os.listdir(model_path):
    
    dataset_path = os.path.join(model_path,dataset)
    if os.path.isdir(dataset_path):
      print('\t', dataset)
      results[model][dataset] = []
      for subpath in os.listdir(dataset_path):
        submodel_path = os.path.join(dataset_path, subpath)
        if os.path.isdir(submodel_path) and os.path.exists(os.path.join(submodel_path, 'classification_report.json')):
          with open(os.path.join(submodel_path, 'classification_report.json')) as f:
            results[model][dataset].append(json.load(f)['accuracy'])
      print('\t\t', np.mean(results[model][dataset]), '+/-', np.std(results[model][dataset]))

#print(results)
