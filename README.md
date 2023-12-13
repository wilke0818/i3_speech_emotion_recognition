
# Exploring the Impact of Model Architectures, Language-based Pre-Fine-Tuning, and Test Datasets on Speech Emotion Recognition

This GitHub repository contains code and resources for the paper titled "Exploring the Impact of Model Architectures, Language-based Pre-Fine-Tuning, and Test Datasets on Speech Emotion Recognition", presented at ICASSP 2024. In this project, we explore the use of self-supervised models for speech emotion recognition, evaluating the results through statistical analyses. 


## Paper's abstract
Speech Emotion Recognition (SER) is being increasingly applied in many societal contexts, often without adequate benchmarking. This study investigates the effects of model architectures, language-based pre-fine-tuning, and test datasets on the accuracy of SER systems, providing valuable insights for future SER studies and applications. 
We ran a statistical evaluation on two Italian emotional speech datasets, i.e., Emozionalmente and EMOVO, employing two distinct self-supervised model architectures (Wav2vec 2.0 and HuBERT), both with and without Italian pre-fine-tuning. 
We found that model architectures and test datasets individually wield significant influence over the accuracy scores. Emozionalmente outperforms EMOVO with a highly significant difference, making a strong case for using Emozionalmente as an Italian SER dataset, and Wav2vec 2.0 shows a similar level of significance in favor of HuBERT. 
Also, we found that model architectures, language-based pre-fine-tuning, and test datasets exhibit complex and interdependent interactions that collectively impact the accuracy of SER systems.

## Repository Structure
This repository is organized as follows: 
- ```requirements.txt```: A file containing the required Python dependencies for running the code in the notebook.
- ```LICENSE```: The license file governing the use and distribution of the code and resources in this repository.
- ```README.md```: The readme file you are currently reading.
- ```...```: ... 

#### TO DO:
- [ ] ([Jordan](https://github.com/wilke0818)) Mention here where are the scripts and where are the supplementary resources (e.g., confusion matrices)
 
## Getting Started
### (Optional) Creating a virtual environment
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

### Package dependencies
From the main project folder run the following command to download all of the packages that you will need + some that you may want for experimentation. 
```
pip install -r requirements.txt
```
Note that you might already have torch-audiomentations installed for augmenting the audio data. In our code we use AddColoredNoise which the current code of torch-audiomentations on PyPi does not fully match their Github. For this reason we reccomend uninstalling it and using the code directly from their Github which is what our requiments.txt file uses.

#### TO DO:
- [ ] ([Jordan](https://github.com/wilke0818)) Explain in detail what .py files should be run to reproduce the ICASSP pipeline. 

## Citation
Hopefully, a paper will be published soon :)

## Questions and Issues
If you have any questions or encounter any issues while using this repository, please feel free [to open an issue](https://github.com/wilke0818/i3_speech_emotion_recognition/issues). We are here to assist you.

Thank you for your interest in our research, and we hope this repository proves valuable in your exploration of speech emotion recognition through self-supervised learning models.


