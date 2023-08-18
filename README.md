
# Getting started

## (Optional) Creating a virtual environment
We recommend using a virtual environment to help manage your package and versions. This most commonly is done with [MiniConda](https://docs.conda.io/en/latest/miniconda.html). Once you have the correct version installed and ready you can create an environment by doing the following:
```
conda create -n ser
conda activate ser
conda install python=3.8
```
All packages and training were done with Python 3.8 but we have seen that the code does work with newer versions of Python.

At any time you can get out of your virtual environment by running:
```
conda deactivate
```

## Package dependencies
From the main project folder run the following command to download all of the packages that you will need + some that you may want for experimentation. 
```
pip install -r requirements.txt
```
Note that you might already have torch-audiomentations installed for augmenting the audio data. In our code we use AddColoredNoise which the current code of torch-audiomentations on PyPi does not fully match their Github. For this reason we reccomend uninstalling it and using the code directly from their Github which is what our requiments.txt file uses.
