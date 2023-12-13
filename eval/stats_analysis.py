import numpy as np
import os
import json
from scipy.stats import mannwhitneyu

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
      results[model][dataset] = {}
      for subpath in os.listdir(dataset_path):
        submodel_path = os.path.join(dataset_path, subpath)
        if os.path.isdir(submodel_path) and os.path.exists(os.path.join(submodel_path, 'accuracies.json')):
          with open(os.path.join(submodel_path, 'accuracies.json')) as f:
            acc = json.load(f)
            for div, div_acc in acc.items():
              results[model][dataset].setdefault(div,[])
              results[model][dataset][div].append(div_acc*100) 
            #results[model][dataset].append(json.load(f)['accuracy'])
      gender_diff = [y for x,y in results[model][dataset].items() if x!='overall']
      for div in results[model][dataset]:
        print('\t\t', div)
        print('\t\t\t', round(np.mean(results[model][dataset][div]),2), '+/-', round(np.std(results[model][dataset][div]),2), 'n=', len(results[model][dataset][div]))
      if len(gender_diff)==2:
        print('\t\t','Testing if the genders are significantly different:', mannwhitneyu(gender_diff[0],gender_diff[1]))

baseline = results['facebook_wav2vec2-large-xlsr-53_emozionalmente_default']['crema_d']['overall']
for model in results:
    print(model)
    datasets = [x for x in results[model].keys() if x!='emovo' and x!='emotionalmente' and 'reg' not in x]
    for dataset in datasets:
        print('\t',dataset)
        print('\t\t','Testing if this is greater than the baseline', mannwhitneyu(results[model][dataset]['overall'],baseline, alternative='greater'))
        print('\t\t','Testing if this is less than the baseline', mannwhitneyu(results[model][dataset]['overall'],baseline, alternative='less'))

print()
print(mannwhitneyu(results['facebook_wav2vec2-large-xlsr-53_crema_vc_gender_resample']['crema_d_vc_gender']['overall'], results['facebook_wav2vec2-large-xlsr-53_crema_vc_resample']['crema_d_vc']['overall'], alternative='greater'))
#print(results)
