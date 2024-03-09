import json
from cca_cka_embeddings_analysis import *
import os, sys, argparse
import numpy as np
def embeddings_analysis(embeddings_base_path, output_path):
  
  #run through each checkpoint
  #  for each layer of that checkpoint
  #    comparison of it to every other layer in the checkpoint
  #    comparison of it to the same layer in every other checkpoint
  #    linear classifier
  # File structure example: /om2/scratch/Tue/fabiocat/MSP/embeddings_outputs/paper2_crema_2_22/crema_test/0/paper2_wavlm_large/final/layer_x/audio_embeddingss
  # experiment_outputs/seed/model_name/checkpoint_name/layer_number/audio_embeddings
  last_audio_embeddings = None
  seeds = os.listdir(embeddings_base_path)
  embeddings_dict = {}
  
  for seed in seeds:
    embeddings_dict[seed] = {}
    temp_path = os.path.join(embeddings_base_path, seed)
    models = os.listdir(temp_path)
    for model in models:
      embeddings_dict[seed][model] = {}
      temp_path = os.path.join(embeddings_base_path, seed, model)
      checkpoints = os.listdir(temp_path)
      for checkpoint in checkpoints:
        embeddings_dict[seed][model][checkpoint] = {}
        temp_path = os.path.join(embeddings_base_path, seed, model, checkpoint)
        layers = os.listdir(temp_path)
        for layer in layers:
          layer_num = int(layer.replace('layer_', ''))
          embeddings_dict[seed][model][checkpoint][layer_num] = {}
          temp_path = os.path.join(embeddings_base_path, seed, model, checkpoint, layer)
          audio_embeddings = os.listdir(temp_path)
          audio_embeddings.sort()
          assert not last_audio_embeddings or last_audio_embeddings == audio_embeddings
          model_embeddings = []
          for embedding_name in audio_embeddings:
            temp_path = os.path.join(embeddings_base_path, seed, model, checkpoint, layer, embedding_name)
            embedding = np.load(temp_path)
            model_embeddings.append(np.squeeze(embedding))
          model_embeddings = np.array(model_embeddings)
          last_audio_embeddings = audio_embeddings
          print(model_embeddings.shape)
          embeddings_dict[seed][model][checkpoint][layer_num] = {'embeddings': model_embeddings} #going to add at this dict {comp_checkpoint: {comp_layer: cka/cca...}}
  print('got embeddings')
  cca_cka_checkpoint_layer_comp = {}
  #Currently not supporting cross model comparisons
  for seed in embeddings_dict:
    cca_cka_checkpoint_layer_comp[seed] = {}
    #Assume models are the same for each seed (sensible within the context of a single experiment)
    for model in embeddings_dict[seed]:
      cca_cka_checkpoint_layer_comp[seed][model] = {}
      comp_checkpoints = list(embeddings_dict[seed][model].keys())
      for checkpoint in embeddings_dict[seed][model]:
        cca_cka_checkpoint_layer_comp[seed][model][checkpoint] = {}
        for comp_checkpoint in comp_checkpoints:
          cca_cka_checkpoint_layer_comp[seed][model][checkpoint][comp_checkpoint] = {}
          #if checkpoint == comp_checkpoint:
          for layer in embeddings_dict[seed][model][checkpoint]:
              comp_layers = list(embeddings_dict[seed][model][comp_checkpoint].keys())
              cca_cka_checkpoint_layer_comp[seed][model][checkpoint][comp_checkpoint][layer] = {}
              for comp_layer in comp_layers:
                #for embedding in embeddings_dict[seed][model][checkpoint][layer]:
                #run whatever comparison between layers we want
                cka_examples, cka_features, cca_embeddings = run_cka_cca(embeddings_dict[seed][model][checkpoint][layer]['embeddings'], embeddings_dict[seed][model][comp_checkpoint][comp_layer]['embeddings'])
                cca_cka_checkpoint_layer_comp[seed][model][checkpoint][comp_checkpoint][layer][comp_layer] = {'cka_examples': float(cka_examples), 'cka_features': float(cka_features), 'cca_embeddings': float(cca_embeddings)}

  print(cca_cka_checkpoint_layer_comp)
  json_embeddings_dict = embeddings_dict
  for seed in embeddings_dict:
    for model in embeddings_dict[seed]:
      for checkpoint in embeddings_dict[seed][model]:
        for layer in embeddings_dict[seed][model][checkpoint]:
          json_embeddings_dict[seed][model][checkpoint][layer].pop('embeddings')
  with open(output_path, 'w') as f:
    json.dump(cca_cka_checkpoint_layer_comp, f)

if __name__ == "__main__":
  parser=argparse.ArgumentParser()

  parser.add_argument("--experiment_path", help="Path to experiment folder with the embeddings")
  parser.add_argument("--output_path", help="Path to output the model embeddings. should be JSON file")
  args=parser.parse_args()
  embeddings_analysis(args.experiment_path, args.output_path)
