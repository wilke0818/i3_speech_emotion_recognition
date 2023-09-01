from audiomentations import LoudnessNormalization, Compose
import argparse, sys
import torchaudio
import torch
import os
from tqdm import tqdm
from TTS.api import TTS
import random
from utils import apply_func_to_all_wavs
from datasets import Dataset, load_dataset, concatenate_datasets

parser=argparse.ArgumentParser()

parser.add_argument("--output_path", help="Path of audio files once cloned")
parser.add_argument("--input_path", help="Path of audio files to clone")
parser.add_argument("--gender", help="Gender to use")
args=parser.parse_args()

dataset = load_dataset("mozilla-foundation/common_voice_13_0", "it")
setup = False
if setup:
  data = concatenate_datasets([dataset['train'],dataset['validation']])
  sentences = data.filter(lambda x: len(x['sentence'].split(' ')) > 25)

  sentences.to_csv('long_sentences.csv', sep="\t")

data = load_dataset('csv', delimiter='\t', data_files='long_sentences.csv', split='train')
male_data = data.filter(lambda x: x['gender']=='male')
female_data = data.filter(lambda x: x['gender']=='female')
#male_data = load_dataset('csv', delimiter='\t',data_files='long_male_sentences.csv', split='train')
#female_data = load_dataset('csv', delimiter='\t',data_files='long_female_sentences.csv', split='train')
print(male_data)
print(female_data)
all_data = concatenate_datasets([male_data, female_data])

gender = ''
if args.gender:
  gender = args.gender.lower()

if gender == 'male' or gender == 'm':
  speaker_ind = random.randrange(len(male_data))
  speaker = male_data[speaker_ind]
elif gender == 'female' or gender =='f':
  speaker_ind = random.randrange(len(female_data))
  speaker = female_data[speaker_ind]
else:
  speaker_ind = random.randrange(len(all_data))
  speaker = all_data[speaker_ind]

if not args.output_path or not args.input_path:
  raise("Need --input_path and --output_path to be specified")

if not os.path.exists(args.output_path):
  os.makedirs(args.output_path)


vc = TTS("voice_conversion_models/multilingual/vctk/freevc24", gpu=torch.cuda.is_available())

def create_realistic_voice_clones(input_file, output_file):
  vc.voice_conversion_to_file(source_wav=input_file, target_wav=speaker['path'], file_path=output_file)
apply_func_to_all_wavs(args.input_path, args.output_path, create_realistic_voice_clones)


