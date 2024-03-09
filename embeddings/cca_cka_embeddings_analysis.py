import sys, argparse, os
import numpy as np
sys.path.insert(0, './eval')
from cka_cca_analysis import *

#this is a lazy function now because it is more efficient to get embeddings outside of here and reuse them
def run_cka_cca(model1_embeddings, model2_embeddings):

  cka_from_examples = cka(gram_linear(model1_embeddings), gram_linear(model2_embeddings))
  cka_from_features = feature_space_linear_cka(model1_embeddings, model2_embeddings)
  cca_from_embeddings = cca(model1_embeddings, model2_embeddings)
  return cka_from_examples, cka_from_features, cca_from_embeddings

def main():
  parser=argparse.ArgumentParser()

  parser.add_argument("--csv_file", help="A CSV file containing a path column to audio files you want the embeddings for")
  parser.add_argument("--input_path", help="Path to .wav files. Takes precedence over --csv_file")
  parser.add_argument("--seed", help="Seed that used to train the model.")
  parser.add_argument("--output_path", help="Path to output the model embeddings")
  args=parser.parse_args()


if __name__ == "__main__":
    main()
