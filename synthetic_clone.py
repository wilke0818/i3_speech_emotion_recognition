from audiomentations import LoudnessNormalization, Compose
import argparse, sys
import torchaudio
import torch
import os
from tqdm import tqdm
from TTS.api import TTS
import random
#from utils import apply_func_to_all_wavs


parser=argparse.ArgumentParser()

parser.add_argument("--output_path", help="Path of audio files once cloned")
parser.add_argument("--input_path", help="Path of audio files to clone")
parser.add_argument("--gender", help="Gender to use")
args=parser.parse_args()

male_speakers = [
  'it_speaker_0',
  'it_speaker_1',
  'it_speaker_3',
  'it_speaker_4',
  'it_speaker_5',
  'it_speaker_6',
  'it_speaker_8'
]
female_speakers = [
  'it_speaker_2',
  'it_speaker_7',
  'it_speaker_9'
]
speaker = ''
gender = ''
if args.gender:
  gender = args.gender.lower()

if gender == 'male' or gender == 'm':
  speaker = random.choice(male_speakers)
elif gender == 'female' or gender =='f':
  speaker = random.choice(female_speakers)
else:
  speaker = random.choice(male_speakers + female_speakers)

if not args.output_path or not args.input_path:
  raise("Need --input_path and --output_path to be specified")

if not os.path.exists(args.output_path):
  os.makedirs(args.output_path)


def apply_func_to_all_wavs(input_path, output_path, func):
  files = os.listdir(input_path)
  for file in files:
    input_file = os.path.join(input_path, file)
    output_file = os.path.join(output_path, file)
    if os.path.isdir(input_file):
      if not os.path.exists(output_file):
        os.makedirs(output_file)
      apply_func_to_all_wavs(input_file, output_file, func)
    elif file.endswith('.wav'):
      func(input_file, output_file)

tts = TTS("tts_models/multilingual/multi-dataset/bark", gpu=torch.cuda.is_available())

#speaker = 'it_speaker_3'
tmp_folder = f'./tmp/{speaker}'

if not os.path.exists(tmp_folder):
  os.makedirs(tmp_folder)
tmp_file = f'{tmp_folder}/output.wav'

tts.tts_to_file(text="Entrambe queste leggende non sono supportate da memorie storiche e non sono molto credibili.",
                file_path=f'{tmp_file}',
                voice_dir="bark_voices/",
                speaker=speaker)
vc = TTS("voice_conversion_models/multilingual/vctk/freevc24", gpu=torch.cuda.is_available())

def create_synthetic_voices(input_file, output_file):
  vc.voice_conversion_to_file(source_wav=input_file, target_wav=tmp_file, file_path=output_file)
apply_func_to_all_wavs(args.input_path, args.output_path, create_synthetic_voices)


