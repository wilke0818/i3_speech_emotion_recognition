
# Exploring the Impact of Model Architectures, Language-based Pre-Fine-Tuning, and Test Datasets on Speech Emotion Recognition

This GitHub repository contains code and resources for the paper titled "Exploring the Impact of Model Architectures, Language-based Pre-Fine-Tuning, and Test Datasets on Speech Emotion Recognition", presented at ICASSP 2024 (sadly we were rejected from ICASSP, to be updated when we figure out where to go from here). In this project, we explore the use of self-supervised models for speech emotion recognition, evaluating the results through statistical analyses. 

[A demo exploring Speech Emotion Recognition that attempts to classify your own voice (no data is saved or used from this)](https://huggingface.co/spaces/wilke18/ItalianSER) - TODO: weird errors started popping up :'(

##Interspeech 2024 Papers (under review)
[What Happens To WavLM Embeddings During Speech Emotion Recognition Fine-tuning?](https://github.com/wilke0818/i3_speech_emotion_recognition/blob/main/interspeech_model_embeddings.pdf)
[Exploring the Impact of Model Architectures, Language-based Pre-finetuning, and Test Datasets on Speech Emotion Recognition](https://github.com/wilke0818/i3_speech_emotion_recognition/blob/main/interspeech_model_architectures.pdf) - We have edited the original paper based on feedback from the ICASSP reviewers and have resubmitted it for consideration.

## Oringal ICASSP Paper's abstract
Speech Emotion Recognition (SER) is being increasingly applied in many societal contexts, often without adequate benchmarking. This study investigates the effects of model architectures, language-based pre-fine-tuning, and test datasets on the accuracy of SER systems, providing valuable insights for future SER studies and applications. 
We ran a statistical evaluation on two Italian emotional speech datasets, i.e., Emozionalmente and EMOVO, employing two distinct self-supervised model architectures (Wav2vec 2.0 and HuBERT), both with and without Italian pre-fine-tuning. 
We found that model architectures and test datasets individually wield significant influence over the accuracy scores. Emozionalmente outperforms EMOVO with a highly significant difference, making a strong case for using Emozionalmente as an Italian SER dataset, and Wav2vec 2.0 shows a similar level of significance in favor of HuBERT. 
Also, we found that model architectures, language-based pre-fine-tuning, and test datasets exhibit complex and interdependent interactions that collectively impact the accuracy of SER systems.

## Repository Structure
This repository is organized as follows: 
- ```requirements.txt```: A file containing the required Python dependencies for running the code in the notebook.
- ```LICENSE```: The license file governing the use and distribution of the code and resources in this repository.
- ```README.md```: The readme file you are currently reading.
- ```train.py```: Main file for training models. Detailed more explicitly below
- ```setup.py```: File for setting up experiments. Downloads the [Emozionalmente](https://zenodo.org/records/6569824) dataset (and the [EMOVO](https://aclanthology.org/L14-1478/) if desired by passing in ```--include_emovo 0```) and does the prerequisite processing of the data that the model expects as input
- ```experiment_input.py```: File defining how experiment input .json files should look and parameters that can be used to customize the experiments to run.
- ```augmentations/```: Directory used for generating augmentations to apply to the dataset which can then be applied to experiments at run time through ```experiment_input.py```
- ```data/```: Data files used for training and testing models. Files generated from running ```setup.py``` are placed here. Additionally contains `volume_normalize.py` which is a specific augmentation that normalizes the audio but this allows that audio to be saved so it doesn't have to be done every run.
- ```eval/```: Directory of files for evaluating the outputted model with the main file being ```eval/eval_dataset.py``` which takes in a CSV which training outputs
- ```experiments/```: Directory containing files described by ```experiment_input.py```
- ```model/inputs/```: Directory contianing files describing individual models to run per experiment as well as the definition python file describing how to configure such files.
- ```utils/```: Utility files used for training
- ```style_tts_vc.py```: An attempt at voice conversion for a related work
- ```voice_cloning/```: Related output for said work
- Remaining files are related to our specific implementation that we used 

 
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

### Setup:

Begin by running 
```
python setup.py
```
and optionally including ```--include_emovo 1``` in order to download the EMOVO dataset as well (used for cross-corpora testing of generalizability). This should load and process the Emozionalmente dataset into `data/audio4analysis/` (with EMOVO being placed into `./data/EMOVO/` if downloaded).

### (Optional) Normalize the audio for training

Our tests were done by using a normalized set of audio from Emozionalmente and EMOVO, however previous tests showed that this had no effects on the results and so we decided to continue to use it in order to minimize confounds, especially for future systems used "in the wild".
```
python data/volume_normalize.py --input_path /path/to/input --output_path /where/normalized/audio/goes
```
An example run for both EMOVO and Emozionalmente are given below:
```
python data/volume_normalize.py --input_path data/audio4analysis --output_path data/audio4analysis_norm
```
```
python data/volume_normalize.py --input_path data/EMOVO --output_path data/EMOVO_norm
```

Note the code takes the highest root path of the audio files and will then recursively search any subdirectories for .wav files. The ouput directory will be created, if it does not already exist, and will have the same structure as the input directory. **NOTE: you will need to generate a new metadata.csv file that contains the correct pathing if you want to use that normalized audio for training a model.**

## Training
### Experiment configuration

The experiment configuration is very important because it specifies many of the values that are needed for training, specifically where results and intermediate artifacts should go. Many of the experiment configuration options themselves have their own options that we detail below. 

experiment_file:
```
{
    "experiment_name": str #A name for the experiment (used for some folder creation and allows to parallelize runs)
    "output_path": str #Where intermediate model outputs are stored
    "experiment_results_output_path": str #Where to store the final model adn the outputted CSV file that training outputs which will also be where evaluation outputs are stored
    "model_files": str #Path to the folder containing the model input jsons to run or to a specific model json if you only want to run one
    "cross_validation": Optional[int] # How many seeds to run
    "augmentations": Optional[Dict[str, Dict[str, Union[str, int, float]]]] #The specific augmentations to apply to the dataset
    "datasets": Optional[List[Dict[str, str]]] #The datasets to evaluate the trained model on which are listed in the CSV that training outputs
    "hp_amount_of_training_data": Optional[float] #Value between 0 and 1 specifying how much data to use for a hyper-parameter search
    "hp_num_trials": Optional[int] # Number of hyper-parameter searches to perform
    "training_dataset": Optional[Dict[str, str] ] #Similar to the datasets parameter but specifies the dataset to perform training on
}
```

augmentations (see [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations/tree/main) and [audiomentations](https://github.com/iver56/audiomentations)): 
```
{
    "augmentation_name" { #An arbitrary name for an augmentation but allows for multiple of the same type of augmentation
        "module": str: #name of the module where the augmentation can be found (or the file path to a custom augmentation),
        "class": str #The class of augmentation to use. Must be a class in the above module,
        "tensors": Optional[bool] #Defaults to false. Whether the augmentation takes tensors as inputs
        #all further parameters are those required or accepted by the specified augmentation class within the module
    }
}
```

datasets (list) 
```
[{
    "name": str #Name of the dataset. Used for folder outputs,
    "eval_csv_path": str #Path to the dataset CSV (but not the CSV itself) that gives the paths of all files to test on as well as their true value. The format of which is listed below but unlike the metadata CSVs, must be named test.csv
}]
```

training_dataset (single dictionary):
```
{
    "name": str #Name of the dataset. Used for folder outputs,
    "data_csv": str #Path to the dataset CSV (the CSV itself) that gives the paths of all files to test on as well as their true value. Format is listed below but unlike the eval_csv_path above, this file may be named anything.
}
```

data_csv (comma separated) and the CSV file at eval_csv_path (tab separated):
```
path,class_id,actor,gender
```
Currently most of our support is for speaker independent Speech Emotion Recognition and as such we need to identify who the actors are. Gender is currently required to keep the proportion of each gender in the train and test set the same as in the entire dataset.


### Model configuration
For each model that you want to train and test you need a model configuration file. This can be a path to many model training files or the exact path to a single JSON.

```
{
    "model_path_or_name": str #Used to grab the pretrained model
    "name": Optional[str] #Name that's helpful for saving files and intermediate artifacts
    "batch_size": Optional[int]
    "per_device_batch_size": Optional[int]
    "pooling_mode": Optional[str] #The type of aggregation to do on the classification head
    "speaker_independent_scenario": Optional[bool] #Whether a speaker in the training set can be in the test set
    "eval_steps": Optional[int] #Number of steps in between evaluations of the model during training
    "logging_steps": Optional[int] #Number of steps between logging progress of the model training
    "input_column": Optional[str] #The input column in data_csv
    "output_column": Optional[str] #The output column in data_csv
    "is_regression": Optional[bool] #Whether we are performing a regression problem
    "train_test_split": Optional[float] #Decimal amount of data in train-test-split. Note train will be N^2, validation N*(1-N), and test N in terms of the proportion of the amount of data
    "seed": Optional[int] # Seed to use
    "use_batch_norm": Optional[bool] #Whether to batch norm in the classification
    "use_dropout": Optional[bool] #Whether to use dropout in the classification
    "dropout_rate": Optional[float] 
    "use_l2_reg": Optional[bool] #Whether to use the L2 regularization
    "weight_decay": Optional[float] #weight decay for L2 regularization
    "number_of_training_epochs": Optional[int]
    "continue_model_training": Optional[bool] = False #if True, model output directory must have the last saved checkpoints of the training
    "skip_hp_search": Optional[bool] #Determine whether to do hyperparameter fine-tuning
}
```

### Run training
If you followed the above steps, you should now be ready to train a model:
```
python train.py --experiment_file /path/to/experiment/config/json
```
Additionally, two optional parameters are `low_seed` and `high_seed` which can be used to specify a range of seeds to run (helped with parallelizing training runs). If both are specified they take precedence over `cross_validation` from the experiment configuration file.

## Evaluation
Congrats on successfully training a model! Now to figure out how well it does.

### Evaluating predicted class_id's
After training finished the model should have outputted a CSV (though it is tab separated) file in the specified location. This file is dated by epoch time. Below are the column names in the evaluation file in case you have trained your model separately and just want to use the evaluation script.
```
model_name      dataset_name    model_path      eval_csv_path   eval_out_path
```
Every row of this CSV is an evaluation. Since a model could have N seeds that were trained, then there should be number of models*N*number of datasets (including the training dataset) evaluations. The `model_name` and `dataset_name` are used for keeping evaluations of the same model/dataset pairing togethe 

Now you can run an evaluation:
```
python eval/eval_dataset.py --eval_path /path/to/generated/eval/csv
```

This will evaluate every model listed in the CSV on the specific model/dataset pair. The code calculates the unweighted average for every model type on each dataset for each seed and will then average them together to get an average unweighted accuracy of the given model for the parameters that were chosen. Additionally, this code generates confusion matrices to visualize per class performance for a given model on a dataset.

### Creating training graphs
Part of evaluating a model might be to get a better understanding of how well it trained in order to see if it is over or under fitting. To accomplish this we provide code that takes the logs of each model and creates a graph of the training accuracy, validation accuracy, and the training loss.

```
python eval/graph_analysis.py --input_path /path/to/model/logs/ --output_path /path/to/save/graph
```
If the `output_path` is not specified then the graph will attempt to be shown. The `input_path` is not directly one of the intermediate artifacts paths created during training but should rather be a directory above that. This means that you will likely need to make such a directory unless one exists already and move the intermediate artifacts into this directory. This is meant to allow you to plot multiple models in comparison with one another.

### Getting quick stats
The evaluation process generates a lot of data and so rather than combing through all of that we provide a simple script for garnering all of the accuracies on every seed for each model on each data set and prints out the mean and standard deviations of the accuracies. There is also currently some code for investigating gender differences in the models to see if it learned to classify one better than another.

## Citation
Hopefully, a paper will be published soon (sadly we were rejected from ICASSP but more work is being done and we will continue to update and extend this repo).

## Questions and Issues
If you have any questions or encounter any issues while using this repository, please feel free [to open an issue](https://github.com/wilke0818/i3_speech_emotion_recognition/issues). We are here to assist you.

Thank you for your interest in our research, and we hope this repository proves valuable in your exploration of speech emotion recognition through self-supervised learning models.


