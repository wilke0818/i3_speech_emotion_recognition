from audiomentations import LoudnessNormalization, Compose
import argparse, sys
import torchaudio
import torch
import os
from tqdm import tqdm
from utils import apply_func_to_all_wavs

parser=argparse.ArgumentParser()

parser.add_argument("--output_path", help="Path of audio files once normalized")
parser.add_argument("--input_path", help="Path of audio files to normalize")

args=parser.parse_args()

if not args.output_path or not args.input_path:
  raise("Need --input_path and --output_path to be specified")

if not os.path.exists(args.output_path):
  os.makedirs(args.output_path)

augment = Compose([
  LoudnessNormalization(min_lufs=-23, max_lufs=-23, p=1)
])

def normalize(input_file, output_file):
  input_audio, sampling_rate = torchaudio.load(input_file)
  output_audio_np = augment(input_audio.numpy(), sampling_rate)
  output_audio = torch.from_numpy(output_audio_np)
  torchaudio.save(output_file, output_audio, sampling_rate)

apply_func_to_all_wavs(args.input_path, args.output_path, normalize)
'''
files = os.listdir(args.input_path)
for file in tqdm(files):
  input_file = os.path.join(args.input_path, file)
  output_file = os.path.join(args.output_path, file)
  input_audio, sampling_rate = torchaudio.load(input_file)
  
  output_audio_np = augment(input_audio.numpy(), sampling_rate)
  output_audio = torch.from_numpy(output_audio_np)
  torchaudio.save(output_file, output_audio, sampling_rate)
'''
