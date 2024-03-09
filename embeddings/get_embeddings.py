import sys, argparse, os
import torch
import torchaudio
from transformers import AutoModel, AutoConfig, AutoFeatureExtractor
from tqdm import tqdm
import numpy as np
sys.path.insert(0, './utils')
from model_classes import *
import csv
import opensmile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Can generate embeddings from either a csv_path (select wavs) or from an input_path (all wav files in that path)
def get_embeddings_for_model(output_path, model_path, input_path=None, csv_path=None):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    processor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True) #AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = ModelForSpeechClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=config, local_files_only=True).to(device)

    if not input_path and not csv_path:
      raise("input_path or csv_path must be specified")
    elif input_path and csv_path:
      print("Cannot use both csv_path and input_path. Defaulting to input_path")

    audios_path = input_path if input_path else csv_path
    for layer in range(config.num_hidden_layers+1):
      output_layer_path = os.path.join(output_path, f'layer_{layer}')
      get_embeddings_at_layer(output_layer_path, model, config, processor, layer, audios_path)

# higher level function that navigates the model_paths
# for a model:
#   for layer in model:
#     generate embeddings for all audios -> apply_func_to_all_wavs(input_path, output_path=output_folder/model/layer_x/audio_name.npz)


def get_embeddings_at_layer(output_path, model, config, processor, layer, audios_path):
  

  # apply_func_to_all_wavs(wav_files_path, output_path, some_function)
  # some_function:
  # load input_file like below
  # model.pretrained_model(input_data).hidden_states[layer].detach().cpu().numpy()
  # save numpy file to output_file (but need to change output_file extension)
  
  
  target_sampling_rate = processor.sampling_rate if processor else 16000
  def save_embeddings_for_audio(input_file, output_file):
    #load audio file
    audio, sr = torchaudio.load(input_file)

    #resample data if needed
    if sr != 16000:
      audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sampling_rate)
    if audio.shape[0] > 1:
      audio = torch.mean(audio, 0)
    else:
      audio = audio.squeeze(0)


    with torch.no_grad():
      audio = audio.unsqueeze(0).to(device)
      model_embeddings = torch.mean(model.pretrained_model(audio, output_hidden_states=True).hidden_states[layer], dim=1).detach().cpu().numpy()

    output_file = output_file.replace('.wav', '.npy')
    with open(output_file, 'wb') as f:
      np.save(f, model_embeddings)
    
  if audios_path[-4:] == '.csv':
    audio_files = []
    with open(audios_path) as f:
      dialect = csv.Sniffer().sniff(f.read(), delimiters=';,\t')
      f.seek(0)
      reader = csv.DictReader(f, dialect=dialect)
      for row in reader:
        audio_files.append(row['path'])
    for audio_file in tqdm(audio_files):
      os.makedirs(output_path, exist_ok=True)
      out_file = os.path.join(output_path,os.path.split(audio_file)[1])
      save_embeddings_for_audio(audio_file, out_file)
  else:
    apply_func_to_all_wavs(wav_files_path, output_path, save_embeddings_for_audio)


def main():
  
  parser=argparse.ArgumentParser()
  parser.add_argument("--csv_file", help="A CSV file containing a path column to audio files you want the embeddings for")
  parser.add_argument("--output_path", help="Path to output the model embeddings")
  args=parser.parse_args()
  audio_files = []

  
  smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
  )
  with open(args.csv_file) as f:
      dialect = csv.Sniffer().sniff(f.read(), delimiters=';,\t')
      f.seek(0)
      reader = csv.DictReader(f, dialect=dialect)
      for row in reader:
        audio_files.append(row['path'])
  for audio_file in tqdm(audio_files):
      os.makedirs(args.output_path, exist_ok=True)
      out_file = os.path.join(args.output_path,os.path.split(audio_file)[1])
      #save_embeddings_for_audio(audio_file, out_file)
  
      
      model_embeddings = smile.process_file(audio_file).to_numpy()
      output_file = out_file.replace('.wav', '.npy')
      with open(output_file, 'wb') as f:
        np.save(f, model_embeddings)

  """
  model_name_or_path = "microsoft/wavlm-large"
  processor = AutoFeatureExtractor.from_pretrained(model_name_or_path,)
  config = AutoConfig.from_pretrained(model_name_or_path)
  setattr(config, 'pooling_mode', 'mean')
  setattr(config, 'use_dropout', False)
  setattr(config, 'dropout_rate', 0)
  setattr(config, 'use_batch_norm', False)
  setattr(config, 'use_l2_reg', False)
  setattr(config, 'weight_decay', 0)
  setattr(config, 'use_weight_encoder_layers', False)
  setattr(config, 'pool_position', 'after')
  setattr(config, 'use_relu', False)


  model = ModelForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config
  ).to('cuda')
  model.freeze_feature_extractor()
  for layer in range(config.num_hidden_layers+1):
    output_layer_path = os.path.join(args.output_path, f'layer_{layer}')
    get_embeddings_at_layer(output_layer_path, model, config, processor, layer, args.csv_file)
  """
  
if __name__ == "__main__":
    main()
