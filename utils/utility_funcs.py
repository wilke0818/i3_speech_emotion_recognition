import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import os
import json

def apply_func_to_all_wavs(input_path, output_path, func):
  files = os.listdir(input_path)
  for file in tqdm(files):
    input_file = os.path.join(input_path, file)
    output_file = os.path.join(output_path, file)
    if os.path.isdir(input_file):
      apply_func_to_all_wavs(input_file, output_file, func)
    elif file.endswith('.wav'):
      if not os.path.exists(output_path):
        os.makedirs(output_path)
      func(input_file, output_file)

