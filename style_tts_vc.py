#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('git clone https://github.com/yl4579/StyleTTS-VC.git')


# In[2]:


get_ipython().run_line_magic('cd', 'StyleTTS-VC/')


# In[3]:


get_ipython().system('pip install SoundFile torchaudio munch torch pydub pyyaml librosa git+https://github.com/resemble-ai/monotonic_align.git')


# In[4]:


get_ipython().system('wget -O model.zip "https://drive.google.com/u/0/uc?id=1bJbj3alOSu51riHUQl4G1GOjlzulyg6M&export=download&confirm=1"')
get_ipython().system('unzip model.zip')
get_ipython().run_line_magic('rm', 'model.zip')
get_ipython().system('wget -O vocoder.zip "https://drive.google.com/u/0/uc?id=1RDxYknrzncGzusYeVeDo38ErNdczzbik&export=download&confirm=1"')
get_ipython().system('unzip vocoder.zip')
get_ipython().run_line_magic('rm', 'vocoder.zip')


# #Start of inference

# In[ ]:


get_ipython().system('pip install datasets')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa

from models import *
from utils import *

get_ipython().run_line_magic('matplotlib', 'inline')



# In[4]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'



to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(ref_dicts):
    reference_embeddings = {}
    for key, path in ref_dicts.items():
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=25)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref = model.style_encoder(mel_tensor.unsqueeze(1))
        reference_embeddings[key] = (ref.squeeze(1), audio)

    return reference_embeddings



# In[7]:


get_ipython().system('pip install addict')


# In[5]:


# load hifi-gan

import sys
sys.path.insert(0, "./Demo/hifi-gan")

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from addict import Dict
from vocoder import Generator
import librosa
import numpy as np
import torchaudio

h = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

cp_g = scan_checkpoint("Vocoder/LibriTTS/", 'g_')

config_file = os.path.join(os.path.split(cp_g)[0], 'config.json')
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = Dict(json_config)

device = torch.device(device)
generator = Generator(h).to(device)

state_dict_g = load_checkpoint(cp_g, device)
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()


# In[6]:


# load StyleTTS
model_path = "./Models/VCTK/epoch_2nd_00100.pth"
model_config_path = "./Models/VCTK/config.yml"

config = yaml.safe_load(open(model_config_path))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

model = build_model(Munch(config['model_params']), text_aligner, pitch_extractor)

params = torch.load(model_path, map_location='cpu')
params = params['net']
for key in model:
    if key in params:
        if not "discriminator" in key:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key])
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]



# In[7]:


get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('cd', '..')


# In[8]:


import argparse, sys
import torchaudio
import torch
import os
from tqdm import tqdm
import random
from datasets import Dataset, load_dataset, concatenate_datasets

setup = 'long_sentences.csv' not in os.listdir('./voice_cloning/')
print(setup)

try:
  dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en")
except OSError as e:
  print(e)
  get_ipython().system('huggingface-cli login')
  dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en")
if setup:

  data = concatenate_datasets([dataset['train'],dataset['validation']])
  sentences = data.filter(lambda x: len(x['sentence'].split(' ')) > 25)

  sentences.to_csv('voice_cloning/long_sentences.csv', sep="\t")

data = load_dataset('csv', delimiter='\t', data_files='voice_cloning/long_sentences.csv', split='train')
male_data = data.filter(lambda x: x['gender']=='male')
female_data = data.filter(lambda x: x['gender']=='female')
#male_data = load_dataset('csv', delimiter='\t',data_files='long_male_sentences.csv', split='train')
#female_data = load_dataset('csv', delimiter='\t',data_files='long_female_sentences.csv', split='train')
print(male_data)
print(female_data)
all_data = concatenate_datasets([male_data, female_data])


# In[9]:


output_path = '/om2/user/wilke18/CREMA-D/AudioWAV_multiplied/'#'./data/audio4analysis_vc_real'
input_path = '/om2/user/wilke18/CREMA-D/AudioWAV_gender/male'#'./data/audio4analysis'
gender = 'male'


# In[10]:


if gender == 'male' or gender == 'm':
  speaker_ind = 4#random.randrange(len(male_data))
  speaker = male_data[speaker_ind]
elif gender == 'female' or gender =='f':
  speaker_ind = 4#random.randrange(len(female_data))
  speaker = female_data[speaker_ind]
else:
  speaker_ind = random.randrange(len(all_data))
  speaker = all_data[speaker_ind]

if not os.path.exists(output_path):
  os.makedirs(output_path)


# In[34]:


get_ipython().system('pip install pydub')


# In[41]:


get_ipython().system('pip install ffmpeg')
get_ipython().system('conda install -y ffmpeg')


# In[11]:


from pydub import AudioSegment
# get first 3 test sample as references

# test_path = val_path.replace('/val_list.txt', '/test_list.txt')
# _, test_list = get_data_path_list(train_path, test_path)

#Different from StarGAN as this doesn't have a number associated with each speaker in the ref_dicsts
ref_dicts = {}
# for j in range(3):
#     filename = test_list[j].split('|')[0]
#     name = filename.split('/')[-1].replace('.wav', '')
#     ref_dicts[name] = filename
print(speaker)
f = speaker['path']

sound = AudioSegment.from_mp3(f)
sound.export('temp.wav', format="wav")

key = os.path.basename('temp.wav').replace('.wav', '')

ref_dicts[key] = 'temp.wav'

print(ref_dicts)

reference_embeddings = compute_style(ref_dicts)



# In[12]:


# src_path = 'Data/src/'
# converted_dicts = []
# for f in os.listdir(src_path):
#   # get last test sample as input
#   # filename = test_list[-1].split('|')[0]
#   filename = os.path.join(src_path, f)
#   audio, source_sr = librosa.load(filename, sr=24000)
#   audio, index = librosa.effects.trim(audio, top_db=30)
#   audio = audio / np.max(np.abs(audio))
#   audio.dtype = np.float32
#   source = preprocess(audio).to(device)
#   converted = conversion(source, filename, reference_embeddings)
#   converted_dicts.append((converted, audio))



# In[13]:


def conversion(source, source_name, reference_embeddings):
  converted_samples = {}
  with torch.no_grad():
      mel_input_length = torch.LongTensor([source.shape[-1]])
      asr = model.mel_encoder(source)
      F0_real, _, F0 = model.pitch_extractor(source.unsqueeze(1))
      real_norm = log_norm(source.unsqueeze(1)).squeeze(1)

      for key, (ref, _) in reference_embeddings.items():
          out = model.decoder(asr, F0_real.unsqueeze(0), real_norm, ref.squeeze(1))

          c = out.squeeze()
          y_g_hat = generator(c.unsqueeze(0))
          y_out = y_g_hat.squeeze()

          converted_samples[source_name] = y_out.cpu().numpy()
      return converted_samples



# In[14]:


get_ipython().run_line_magic('pwd', '')
print(os.getcwd())
# from data_splitter import generate_dataset


# In[ ]:


sys.path.insert(0, './utils/')
from utility_funcs import apply_func_to_all_wavs



# vc = TTS("voice_conversion_models/multilingual/vctk/freevc24", gpu=torch.cuda.is_available())
target_sr = 24000

def create_realistic_voice_clones(input_file, output_file):
  audio, source_sr = torchaudio.load(input_file)
  resampler = torchaudio.transforms.Resample(source_sr, target_sr)
  audio = resampler(audio).squeeze().numpy()

  audio, index = librosa.effects.trim(audio, top_db=30)
  audio = audio / np.max(np.abs(audio))
  audio.dtype = np.float32
  source = preprocess(audio).to(device)
  converted = conversion(source, input_file, reference_embeddings)
  # print(converted)
  resampler = torchaudio.transforms.Resample(target_sr, source_sr)
  out_audio = resampler(torch.from_numpy(converted[input_file]).unsqueeze(0))
  split = os.path.split(output_file)
  output_file = os.path.join(split[0], f"{gender}_{speaker_ind}_{split[1]}")
  torchaudio.save(output_file, torch.from_numpy(converted[input_file]).unsqueeze(0), target_sr)

apply_func_to_all_wavs(input_path, output_path, create_realistic_voice_clones)


# In[33]:


# import IPython.display as ipd
# for converted_sample, wave2 in converted_dicts:
#   for key, wave in converted_sample.items():
#       print('Converted: %s' % key)
#       display(ipd.Audio(wave, rate=24000))
#       try:
#           print('Reference: %s' % key)
#           display(ipd.Audio(wave2, rate=24000))
#       except:
#           continue

# for key in reference_embeddings:
#   try:
#       print('Original: %s' % key)
#       display(ipd.Audio(reference_embeddings[key][-1], rate=24000))
#   except:
#       continue
# # print('Original:')
# # display(ipd.Audio(audio, rate=24000))


