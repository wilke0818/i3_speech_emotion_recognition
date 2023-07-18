import pandas as pd
import os
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
    #print(type(actor))
    #print(type(actor[0]))
    #print(actor[0])
    directory = os.path.join('.',actor)
    for f in os.listdir(directory):
        with open(os.path.join(directory, f)) as file:
            letter = actor[0]
            #print(letter)
            gender = 'male' if bool(str(actor[0])=="m") else 'female' #no other's in this dataset
            emotion = emotions[f[0:3]]
            person = 3*(actor[0]=='m')+int(actor[1])
            path = os.path.join('.',f'EMOVO/{actor}/{f}')
            data.append({
                'gender': gender,
                'emotion': emotion,
                'actor': str(person),
                'path': path
                })
df = pd.DataFrame(data)

df.to_csv(f"../data/emovo/test.csv", sep="\t", encoding="utf-8", index=False)
