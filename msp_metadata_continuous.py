import os
import json
import argparse
import torchaudio
import csv
import pandas as pd

parser=argparse.ArgumentParser()

parser.add_argument("--output_path", help="Where to put/call the metadata csv")
parser.add_argument("--labels_file", help="Path to the labels csv", default="/om2/scratch/Tue/fabiocat/MSP/Labels/labels_consensus.json")
parser.add_argument("--input_path", help="Path to .wav files referenced in labels_csv")
parser.add_argument("--split_by_set", help="Splits into the sets defined by MSP Podcast", action="store_true")

args=parser.parse_args()


with open(args.labels_file) as f:
  if '.json' in args.labels_file:
    use_csv = False
    audio_metadata = json.load(f)
    audio_iterator = audio_metadata.items()
  else:
    use_csv = True
    audio_metadata = csv.DictReader(f)
    audio_iterator = audio_metadata

data = {'metadata': [], 'Test1': [], 'Test2': [], 'Train': [], 'Development': []}

for audio in audio_iterator:
  file_name = audio['FileName'] if use_csv else audio[0]
  audio = audio if use_csv else audio[1]
  emotion = audio['EmoClass']
  arousal = audio['EmoAct']
  valence = audio['EmoVal']
  dominance = audio['EmoDom']
  #if emotion in ['O', 'X']:
  #  continue
  path = os.path.join(args.input_path, file_name)
  actor = audio['SpkrID']
  gender = audio['Gender']
  age = 0
  try:  
    s = torchaudio.load(path)
    data[audio['Split_Set']].append([path, actor, emotion, arousal, valence, dominance,gender, age])
    data['metadata'].append([path, actor, emotion, arousal, valence, dominance, gender, age])

  except Exception as e:
            # Check if there are some broken files
    print(str(path), e)
    raise(e)


for subclass in data.keys():
  df = pd.DataFrame(data[subclass], columns=['path', 'actor', 'class_id','emo_act', 'emo_val', 'emo_dom', 'gender', 'age']) 
  df.to_csv(os.path.join(args.output_path, f"{subclass.lower()}.csv"))

print('done')
