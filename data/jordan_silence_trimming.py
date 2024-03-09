import argparse, sys
import torchaudio
import torch
import os, sys
from tqdm import tqdm

import glob
from pyannote.audio import Pipeline
import pyloudnorm as pyln

sys.path.insert(0, 'utils')

from utility_funcs import apply_func_to_all_wavs

parser=argparse.ArgumentParser()

parser.add_argument("--output_path", help="Path of audio files once normalized")
parser.add_argument("--input_path", help="Path of audio files to normalize")

args=parser.parse_args()

if not args.output_path or not args.input_path:
  raise("Need --input_path and --output_path to be specified")

if not os.path.exists(args.output_path):
  os.makedirs(args.output_path)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cuda"))

def vad(audio_path):
  output = pipeline(audio_path)
  length = sum(1 for dummy in output.itertracks(yield_label=True))
  if length > 0:
    turn, _, speaker = next(output.itertracks(yield_label=True))
    start = turn.start
    *_, last = output.itertracks(yield_label=True)
    turn, _, speaker = last
    end = turn.end
  #print('start', start, 'end', end)
#  if output.get_timeline().support():
#    start = output.get_timeline().support()[0].start
#    end = output.get_timeline().support()[-1].end
  else:
    start = 0
    end = -1
  return start, end


def trim_audio(audio_path, start_seconds, end_seconds):
  """
  Trims an audio file to the specified start and end times.

  Args:
    audio_path: The path to the audio file.
    start_seconds: The start time in seconds.
    end_seconds: The end time in seconds.

  Returns:
    The trimmed audio as a torchaudio.Audio object.
  """

  # Load the audio file
  audio, sr = torchaudio.load(audio_path)

  # Calculate the start and end samples
  start_samples = int(start_seconds * sr)
  end_samples = int(end_seconds * sr)

  # Trim the audio
  trimmed_audio = audio[:, start_samples:end_samples] if end_seconds != -1 else audio[:, start_samples:]
  return trimmed_audio, sr

def silence_remover(input_file, output_file):
    start, end = vad(input_file)
    trimmed_audio, sr = trim_audio(input_file, start, end)
    torchaudio.save(output_file, trimmed_audio, sr)

#silence_remover('data/audio4analysis/female____2ee38ab6370fe93156f4b8aba1d17e88fb8e5b1fa02264a8f699640601e671bb____anger____1615555236661.wav', 'test.wav')
apply_func_to_all_wavs(args.input_path, args.output_path, silence_remover)
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
