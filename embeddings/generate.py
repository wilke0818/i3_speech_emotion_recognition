import os, sys, argparse
from get_embeddings import *
from tqdm import tqdm

def generate_embeddings_for_experiment_path(experiment_path, input_path, output_path, seed, csv_file):
  #Experiment model saves:
  #experiment_path/seed_or_checkpoints/model_name/...
  all_runs = os.listdir(experiment_path)
  temp_path = experiment_path
  seeds = os.listdir(experiment_path)
  if 'checkpoints' in all_runs:
    all_runs.remove('checkpoints')
    
    if seed==None:
      seeds = all_runs
    else:
      seeds = [seed]
    for seed in seeds:
      temp_path = os.path.join(experiment_path, 'checkpoints', str(seed))
      model_runs = os.listdir(temp_path)
      for model_run in model_runs:
        temp_path = os.path.join(experiment_path, 'checkpoints', str(seed), model_run)
        checkpoints = os.listdir(temp_path)
        for checkpoint in checkpoints:
          temp_path = os.path.join(experiment_path, 'checkpoints', str(seed), model_run, checkpoint)
          embeddings_output_path = os.path.join(output_path, str(seed), model_run, checkpoint.replace('checkpoint-', ''))
          get_embeddings_for_model(output_path=embeddings_output_path, model_path=temp_path, input_path=input_path, csv_path=csv_file)

  for seed in seeds:
    temp_path = os.path.join(experiment_path, str(seed))
    model_runs = os.listdir(temp_path)
    for model_run in tqdm(model_runs):
      temp_path = os.path.join(experiment_path,str(seed), model_run)
      embeddings_output_path = os.path.join(output_path, str(seed), model_run, 'final')
      get_embeddings_for_model(output_path=embeddings_output_path, model_path=temp_path, input_path=input_path, csv_path=csv_file)

def main():
  parser=argparse.ArgumentParser()

  parser.add_argument("--experiment_path", help="Path of the experiment files, notably this will check for checkpoints and final model. Gets embeddings for all model types")
  parser.add_argument("--csv_file", help="A CSV file containing a path column to audio files you want the embeddings for") 
  parser.add_argument("--input_path", help="Path to .wav files. Takes precedence over --csv_file")
  parser.add_argument("--seed", help="Seed that used to train the model.")
  parser.add_argument("--output_path", help="Path to output the model embeddings")
  args=parser.parse_args()
  generate_embeddings_for_experiment_path(args.experiment_path, args.input_path, args.output_path, args.seed, args.csv_file)


if __name__ == "__main__":
    main()
