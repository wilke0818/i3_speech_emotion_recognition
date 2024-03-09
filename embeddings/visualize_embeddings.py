import re
import json
import os, argparse
import matplotlib.pyplot as plt

def sorted_nicely(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def visualize_checkpoint_diffs(embeddings_dict, checkpoint_a, checkpoint_b, output_path):
  comparison_matrix = {}
  for layer_a in embeddings_dict[checkpoint_a][checkpoint_b]:
    comparison_matrix[int(layer_a)] = {}
    for layer_b in embeddings_dict[checkpoint_a][checkpoint_b][layer_a]:
      comparison_matrix[int(layer_a)][int(layer_b)] = embeddings_dict[checkpoint_a][checkpoint_b][layer_a][layer_b]['cka_features']
  
  comp_list = []
  for a in range(len(comparison_matrix)):
    comp_list.append([])
    for b in range(len(comparison_matrix[a])):
      comp_list[a].append(comparison_matrix[a][b])

  y_label = f'Checkpoint {checkpoint_a} Layers'
  x_label = f'Checkpoint {checkpoint_b} Layers'
  y_ticks = range(len(comparison_matrix))
  x_ticks = range(len(comparison_matrix))
  title = f'Checkpoint {checkpoint_a} vs. Checkpoint {checkpoint_b} Layer Comparison'
  output_file = os.path.join(output_path, f'checkpoint_{checkpoint_a}_vs_checkpoint_{checkpoint_b}.png')
  visualize_similarity_matrix(comp_list, output_file, title, x_label, y_label, x_ticks, y_ticks)

def visualize_layer_changes(embeddings_dict, layer_a, layer_b, output_path):
  comparison_matrix = {}
  numeric_checkpoints = []
  for checkpoint_a in embeddings_dict:
    comparison_matrix[checkpoint_a] = {}
    for checkpoint_b in embeddings_dict[checkpoint_a]:
      comparison_matrix[checkpoint_a][checkpoint_b] = embeddings_dict[checkpoint_a][checkpoint_b][layer_a][layer_b]['cka_features']

  comp_list = []
  comp_keys_a = sorted_nicely(list(comparison_matrix.keys()))
  for a in range(len(comp_keys_a)):
    comp_list.append([])
    comp_keys_b = sorted_nicely(list(comparison_matrix[comp_keys_a[a]]))
    for b in range(len(comp_keys_b)):
      comp_list[a].append(comparison_matrix[comp_keys_a[a]][comp_keys_b[b]])

  y_label = f'Layer {layer_a} in Each Checkpoint'
  x_label = f'Layer {layer_b} in Each Checkpoint'
  y_ticks = comp_keys_a
  x_ticks = comp_keys_b
  title = f'Layer {layer_a} vs. {layer_b} Over Training'
  output_file = os.path.join(output_path, f'layer_{layer_a}_vs_layer_{layer_b}.png')
  visualize_similarity_matrix(comp_list, output_file, title, x_label, y_label, x_ticks, y_ticks)

def visualize_egemaps(embeddings_dict, output_path):
  checkpoints = sorted_nicely(list(embeddings_dict['egemaps'].keys()))
  checkpoints.reverse()
  checkpoints.remove('egemaps')
  layers = sorted_nicely(list(embeddings_dict['egemaps']['final']['0'].keys()))
  comp_list = []
  for checkpoint_i in range(len(checkpoints)):
    comp_list.append([])
    for layer_i in range(len(layers)):
      comp_list[checkpoint_i].append(embeddings_dict['egemaps'][checkpoints[checkpoint_i]]['0'][layers[layer_i]]['cka_features'])
  #print(comp_list)
  y_label = f'Checkpoint'
  x_label = f'Layer'
  y_ticks = checkpoints
  x_ticks = layers
  title = f'Comparison to EgemapsV2.0 Features'
  output_file = os.path.join(output_path, f'egemaps.png')
  visualize_similarity_matrix(comp_list, output_file, title, x_label, y_label, x_ticks, y_ticks)


def visualize_similarity_matrix(sim_matrix, output_file, title, x_label, y_label, x_ticks, y_ticks):
  fig, ax = plt.subplots(figsize=(10,10))
  cax = ax.imshow(sim_matrix, interpolation='nearest')
  ax.grid(True)
  fig.colorbar(cax)
  ax.set_xticks(range(len(x_ticks)), x_ticks, rotation=90)
  ax.set_yticks(range(len(y_ticks)), y_ticks)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_title(title)
  plt.tight_layout()
  fig.savefig(output_file)
  plt.close()

def visualize_embeddings(embeddings_dict, output_dir):
  layer_output_dir = os.path.join(output_dir, 'layers_comp')
  checkpoint_output_dir = os.path.join(output_dir, 'checkpoint_comp')
  egemaps_dir = os.path.join(output_dir, 'egemaps')

  os.makedirs(layer_output_dir, exist_ok=True)
  os.makedirs(checkpoint_output_dir, exist_ok=True)
  os.makedirs(egemaps_dir, exist_ok=True)
  '''
  checkpoints = list(embeddings_dict.keys())
  checkpoints.remove('egemaps')

  checkpoints = sorted_nicely(checkpoints)
  print('sorted_checkpoints', checkpoints)
  
  checkpoint_bs = checkpoints.copy()
  for checkpoint_a in checkpoints:
    print('checkpoints', checkpoints)
    print('checkpoint_bs', checkpoint_bs)
    for checkpoint_b in checkpoint_bs:
      print(checkpoint_a, 'vs', checkpoint_b)
      visualize_checkpoint_diffs(embeddings_dict, checkpoint_a, checkpoint_b, checkpoint_output_dir)
    checkpoint_bs.remove(checkpoint_a)
  
  layers = sorted_nicely(list(embeddings_dict[checkpoints[-1]][checkpoints[-1]].keys()))
  layer_bs = layers.copy()
  embeddings_dict_no_ege = embeddings_dict
  
  '''
  visualize_egemaps(embeddings_dict, egemaps_dir) 

  '''
  for checkpoint in embeddings_dict_no_ege:
    embeddings_dict_no_ege[checkpoint].pop('egemaps')
  embeddings_dict_no_ege.pop('egemaps')
  
  print('layers comparison starting')

  for layer_a in layers:
    for layer_b in layer_bs:
      visualize_layer_changes(embeddings_dict_no_ege, layer_a, layer_b, layer_output_dir)
    layer_bs.remove(layer_a)
  '''


def visualize_classification_scores(classification_dict, output_path):
  egemaps_classification = classification_dict['egemaps']['0'][0]['f1score']
  classification_dict.pop('egemaps')
  checkpoints = sorted_nicely(list(classification_dict.keys()))
  #checkpoints.remove('egemaps')

  layers = sorted_nicely(list(classification_dict['final'].keys()))
  comp_list = []
  for checkpoint_i in range(len(checkpoints)):
    layers = sorted_nicely(list(classification_dict[checkpoints[checkpoint_i]].keys()))
    comp_list.append([])
    for layer in layers:
      comp_list[checkpoint_i].append(classification_dict[checkpoints[checkpoint_i]][layer][0]['f1score'])


  y_label = f'F1 Score'
  x_label = f'Layers'
  
  title = f'F1 Score by Layer for Various Checkpoints'
  output_file = os.path.join(output_path, f'classification_by_checkpoint_layers.png')
  #visualize_similarity_matrix(comp_list, output_file, title, x_label, y_label, x_ticks, y_ticks)
  fig, ax = plt.subplots(figsize=(10,10))
  for checkpoint_i in range(len(checkpoints)):
    label = f'Checkpoint {checkpoints[checkpoint_i]}' if checkpoints[checkpoint_i] != 'final' else 'Final Model'
    ax.plot(comp_list[checkpoint_i], label=checkpoints[checkpoint_i])
  
  ax.axhline(y=egemaps_classification, color='grey', label='eGeMAPS f1score', ls='--')
  ax.grid(True)
  
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_title(title)
  ax.legend()

  plt.tight_layout()
  fig.savefig(output_file)
  plt.close()


def main():
  parser=argparse.ArgumentParser()

  parser.add_argument("--embeddings_json", help="Path to embeddings json")
  parser.add_argument("--output_path", help="Path to output the model embedding visualizations")
  parser.add_argument("--classification_json", help="Path to classification json")
  args=parser.parse_args()

  with open(args.embeddings_json) as f:
    embeddings = json.load(f)

  with open(args.classification_json) as f:
    classifications = json.load(f)

  embeddings = embeddings['0']['paper2_wavlm_large']
  classifications = classifications['0']['paper2_wavlm_large']

  visualize_embeddings(embeddings, args.output_path)
  visualize_classification_scores(classifications, args.output_path)


if __name__ == "__main__":
    main()
