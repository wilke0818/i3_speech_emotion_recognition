import pandas as pd
import os
import requests
from zipfile import ZipFile
from tempfile import mkdtemp
import wget
import shutil
import sys, argparse

def run_download_data(download_emozionalmente=True, download_emovo=False):
    #Download the emozionalmente data from our dropbox and setup the expected file structure
    #TODO add EMOVO and make more generalizable
    if not os.path.exists('./data'):
        os.makedirs('./data')

    if download_emozionalmente:
        if not os.path.exists('./data/audio4analysis/'):
            os.makedirs('./data/audio4analysis/')


        metadata_file = wget.download('https://www.dropbox.com/s/gi1iwc3xwwl0a4z/metadata.zip?dl=1')
    
        audio_file = wget.download('https://www.dropbox.com/s/tlbxkdabow9w03i/audio.zip?dl=1')
    
        print('Downloaded the metadata and audio zip files')
        print('Extracting the audio zip file. This may take a few minutes')
        with ZipFile(audio_file, 'r') as zip_ref:
            zip_ref.extractall('./data/')

        print('Successfully extracted the audio zip file')
    
        with ZipFile(metadata_file, 'r') as zip_ref:
            zip_ref.extractall('./data/')

        print('Successfully extracted the metadata zip file')

        os.remove(metadata_file)
        os.remove(audio_file)
        print('removed the zip files')
        print('Generating data files in audio4analysis. This may take a few minutes')
        generate_data_files()

    if download_emovo:
        emovo_file = wget.download('https://drive.google.com/u/0/uc?id=1SUtaKeA-LYnKaD3qv87Y5wYgihJiNJAo&export=download&confirm=1')
    
        print('Downloaded emovo zip')
        print('Extracting the emovo zip file. This may take a few minutes')
        with ZipFile(emovo_file, 'r') as zip_ref:
            zip_ref.extractall('./data/')
        
        os.remove(emovo_file)
        process_emovo_data('./data/EMOVO/')
        print('processed emovo data files to generate a test.csv')

def process_emovo_data(emovo_dir):
  import torchaudio

  actors = ['f1', 'f2', 'f3', 'm1', 'm2', 'm3']

  emotions = {
    'dis': 'disgust',
    'gio': 'joy',
    'neu': 'neutrality',
    'pau': 'fear',
    'rab': 'anger',
    'sor': 'surprise',
    'tri': 'sadness'
  }

  data = []

  for actor in actors:
      directory = os.path.join(emovo_dir,actor)
      for f in os.listdir(directory):
          with open(os.path.join(directory, f)) as file:
              letter = actor[0]
              #print(letter)
              gender = 'male' if bool(str(actor[0])=="m") else 'female' #no other's in this dataset
              emotion = emotions[f[0:3]]
              person = 3*(actor[0]=='m')+int(actor[1])
              path = os.path.join(emovo_dir,f'{actor}/{f}')
              data.append({
                  'gender': gender,
                  'emotion': emotion,
                  'actor': str(person),
                  'path': path
                  })
  df = pd.DataFrame(data)

  df.to_csv(f"{emovo_dir}/test.csv", sep="\t", encoding="utf-8", index=False)

def generate_data_files():
    users_df = pd.read_csv("data/metadata/users.csv")
    samples_df = pd.read_csv("data/metadata/samples.csv")

    samples_df['actor_gender'] = None
    samples_df['actor_age'] = None
    samples_df['new_file_name'] = None
    
    for index, sample in samples_df.iterrows():
      actor = sample['actor']
      file_name = sample['file_name']
      emotion_expressed = sample['emotion_expressed']
      age = users_df[users_df['username'] == actor]['age'].values[0]
      gender = users_df[users_df['username'] == actor]['gender'].values[0]
      new_file_name = gender + '____' + actor + '____' + emotion_expressed + '____' + file_name
      samples_df.iloc[index, samples_df.columns.get_loc('actor_age')] = age
      samples_df.iloc[index, samples_df.columns.get_loc('actor_gender')] = gender
      samples_df.iloc[index, samples_df.columns.get_loc('new_file_name')] = new_file_name
      shutil.copyfile(f'data/audio/{file_name}', f'data/audio4analysis/{new_file_name}')


def main():
  parser=argparse.ArgumentParser()

  parser.add_argument("--only_emovo", type=int, help="Download only emovo, uses 0 or 1")
  parser.add_argument("--include_emovo", type=int, help="Download all data including emovo, uses 0 or 1")

  args=parser.parse_args()
  if args.include_emovo == 1:
      run_download_data(True, True)
  elif args.only_emovo == 1:
      run_download_data(False, True)
  else:
      run_download_data(True, False)

  if args.only_emovo and args.only_emovo != 1 and args.only_emovo != 0:
        print('Expected only 0 and 1 for --only_emovo')
  if args.include_emovo and args.include_emovo != 1 and args.include_emovo != 0:
        print('Expected only 0 and 1 for --include_emovo')


if __name__=="__main__":
  main()

