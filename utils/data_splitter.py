import os
import sys
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import torchaudio
import csv

def my_split(df, data, state, training_amount):
    gss = GroupShuffleSplit(n_splits=1, train_size=training_amount, random_state=state)
    gss.get_n_splits()

    my_real_train_idx = []
    my_real_test_idx = []

    for label in df['class_id'].unique():
        df_data_by_emotion = data[data['class_id'] == label]
        X = df_data_by_emotion
        y = df_data_by_emotion["class_id"]
        groups = df_data_by_emotion['actor']
        print('X',X)
        print('y',y)
        print('groups',groups)

        for train_idx, test_idx in gss.split(X, y, groups):
            my_real_train_idx = np.concatenate([my_real_train_idx, df_data_by_emotion.iloc[train_idx].index])
            my_real_test_idx = np.concatenate([my_real_test_idx, df_data_by_emotion.iloc[test_idx].index])

    return my_real_train_idx, my_real_test_idx



def load_saved_dataset(save_path):
    data_files = {
        "train": save_path + "/train.csv",
        "validation": save_path + "/val.csv",
        "test": save_path + "/test.csv",
    }
    with open(data_files['train']) as f:
      dialect = csv.Sniffer().sniff(f.read(), delimiters=';,\t')
      f.seek(0)
      reader = csv.DictReader(f, dialect=dialect)
    dataset = load_dataset("csv", data_files=data_files, dialect=dialect )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    return train_dataset, eval_dataset, test_dataset



def generate_dataset(seed=0, training_split=.8, speaker_independent_scenario=True, save=True, amount_of_data=1, data_csv='./data/audio4analysis/metadata.csv'):
    data = []
    data_path = os.path.split(data_csv)
    data_path = os.path.join(data_path[0], data_path[1][:-4])
    save_path = os.path.join(data_path, f"train_test_validation/{seed}/speaker_ind_{speaker_independent_scenario}_{int(100*amount_of_data)}_{int(100*training_split)}")
    
    #Code is deterministic so don't redo computation if we don't need to
    if os.path.exists(save_path) and 'train.csv' in os.listdir(save_path):
      print('Using previously generated train.csv')
      return load_saved_dataset(save_path)
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    with open(data_csv, 'r') as f:
      dialect = csv.Sniffer().sniff(f.read(), delimiters=';,\t')
      f.seek(0)
      reader = csv.DictReader(f, dialect=dialect)
      for row in reader:
        actor = row['actor']
        class_id = row['class_id']
        path = row['path']
        gender = row['gender']
        #TODO consider adding age
        try:
          s = torchaudio.load(path)
          data.append({
                "path": path,
                "class_id": class_id,
                "actor": actor,
                "gender": gender
          })
        except Exception as e:
            # Check if there are some broken files
            print(str(path), e)
    df = pd.DataFrame(data)
    df.groupby("class_id").count()[["path"]]
    print(df)
    if speaker_independent_scenario == True:
        males = df[df['gender'].str.lower() == "male"]
        females = df[df['gender'].str.lower() == "female"]

        #Allow for getting a given amount of data rather than just a train/test split
        if amount_of_data != 1 and amount_of_data != 0:
            real_train_idx, real_test_idx = my_split(df,males,seed,amount_of_data)
            males = df.iloc[real_train_idx.astype(int)]

            real_train_idx, real_test_idx = my_split(df,females,seed,amount_of_data)
            females = df.iloc[real_train_idx.astype(int)]



        real_train_idx, real_test_idx = my_split(df,males,seed,training_split)
        males_train_df = df.iloc[real_train_idx.astype(int)]
        males_test_df = df.iloc[real_test_idx.astype(int)]
    
        real_train_idx, real_test_idx = my_split(df,females,seed,training_split)
        females_train_df = df.iloc[real_train_idx.astype(int)]
        females_test_df = df.iloc[real_test_idx.astype(int)]
    
        real_train_idx, real_val_idx = my_split(df,males_train_df,seed,training_split)
        males_train_df = df.iloc[real_train_idx.astype(int)]
        males_val_df = df.iloc[real_val_idx.astype(int)]
    
        real_train_idx, real_val_idx = my_split(df,females_train_df,seed,training_split)
        females_train_df = df.iloc[real_train_idx.astype(int)]
        females_val_df = df.iloc[real_val_idx.astype(int)]
    
        train_df = pd.concat([males_train_df,females_train_df])
    
        test_df = pd.concat([males_test_df,females_test_df])
        val_df = pd.concat([males_val_df,females_val_df])
    else: #TODO implement data splits; right now this code only allows for train/test splits
        train_df, test_df = train_test_split(df, test_size=(1-training_amount), random_state=seed, stratify=df["class_id"])
        train_df, val_df = train_test_split(train_df, test_size=(1-training_amount), random_state=seed, stratify=train_df["class_id"])



    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if not save: #We typically want to save the data, but if not, just return the data
        return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df), Dataset.from_pandas(test_df)


    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
    val_df.to_csv(f"{save_path}/val.csv", sep="\t", encoding="utf-8", index=False)
    return load_saved_dataset(save_path)

