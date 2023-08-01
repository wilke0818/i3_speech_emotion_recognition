import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

import argparse, sys

results = {}

log_path = './logs'
models = os.listdir(log_path)

for model in models:
  model_path = os.path.join(log_path, model)
  if not os.path.isdir(model_path):
    continue
  #print(model)
  results[model] = {}
  for model_run in os.listdir(model_path): 
    model_run_path = os.path.join(model_path,model_run)
    if os.path.isdir(model_run_path):
      #print('\t', dataset)
      results[model][model_run] = {'eval_loss': [], 'eval_accuracy': []}
      
      sub_path = os.path.join(model_run_path, 'runs')
      if len(os.listdir(sub_path)) >= 1:
        all_sub_paths = [os.path.join(sub_path, sub_sub) for sub_sub in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, sub_sub))]
        all_sub_path_times = [datetime.strptime(sub_sub[0:13], '%b%d_%H-%M-%S') for sub_sub in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, sub_sub))]
        #print(all_sub_path_times)
        sorted_all_sub_paths = sorted(all_sub_paths, key=lambda x:all_sub_path_times[all_sub_paths.index(x)])
        #print(all_sub_paths)
        #print(sorted_all_sub_paths)
        sub_path = sorted_all_sub_paths[-1]
        #if model == 'jonatasgrosman_wav2vec2_large_xlsr_53_italian__augmented__union_extended_train':
        #  for i in range(len(all_sub_paths)):
        #    print(all_sub_paths[i], all_sub_path_times[i])
        #  print(sorted_all_sub_paths)
      elif len(os.listdir(sub_path)) == 1:
        sub_path = os.listdir(sub_path)[0]
      else:
        continue
      print(sub_path)
      if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, 'logs.txt')):
        
        with open(os.path.join(sub_path, 'logs.txt')) as f:
          log_lines = f.readlines()
          if model == 'jonatasgrosman_wav2vec2_large_xlsr_53_italian__augmented__union_extended_train':
            print(model_run, log_lines)
          if len(log_lines) == 1: #we need to split the logs into many lines to match new style
            log_lines = [line.replace('\'', '\"')+'}' for line in log_lines[0].split('}')[0:-1]]
          else:
            log_lines = log_lines[0:-1]
          for line in log_lines:
            log_dict = json.loads(line)
            results[model][model_run]['eval_loss'].append(log_dict['eval_loss'])
            results[model][model_run]['eval_accuracy'].append(log_dict['eval_accuracy'])

#print(results)

#Going to look at just the 0th seed
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,10), sharex=True)

best_model_map = {}
for model in results:
  min_eval_loss = 10000
  max_eval_acc = -1
  best_model_run_loss = -1
  best_model_run_acc = -1
  best_model_map[model] = {}
  for model_run in results[model]:
    #model_result_loss = min(results[model][model_run]['eval_loss'])
    #model_result_acc = max(results[model][model_run]['eval_accuracy'])
    model_result_loss = results[model][model_run]['eval_loss'][-1]
    model_result_acc = results[model][model_run]['eval_accuracy'][-1]
    if model_result_loss < min_eval_loss:
      min_eval_loss = model_result_loss
      best_model_run_loss = model_run
    if model_result_acc > max_eval_acc:
      max_eval_acc = model_result_acc
      best_model_run_acc = model_run
  best_model_map[model]['eval_loss'] = best_model_run_loss
  best_model_map[model]['eval_accuracy'] = best_model_run_acc


for model in results:
  #print(model, results[model]['2'])
  ax1.plot(results[model][best_model_map[model]['eval_accuracy']]['eval_accuracy'], label=model)
  ax2.plot(results[model][best_model_map[model]['eval_loss']]['eval_loss'])

print(best_model_map)

ax1.set(title='Eval Accuracy')
ax1.set(ylabel="Percentage")
ax2.set(title='Eval Loss')
ax2.set(ylabel="Cross Entropy Loss")

fig.supxlabel("Evaluation number")

lines = []
labels = []
for ax in fig.axes:
    Line, Label = ax.get_legend_handles_labels()
    # print(Label)
    lines.extend(Line)
    labels.extend(Label)
fig.legend(lines, labels, loc='upper right')
#plt.tight_layout()

parser=argparse.ArgumentParser()

parser.add_argument("--save_path", help="Where to save the graph. If not set then attempts to show graph")


args=parser.parse_args()
print(args)

if args.save_path:
  fig.savefig(args.save_path)
else:
  plt.show()
