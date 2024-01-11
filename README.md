# Evaluation Results
We have evaluated our model on spider dataset.
| Model   | Dev EM | Dev EX | Test EM | Test EX |
|---------|--------|--------|---------|---------|
| SQL-GPT | 84.3   | 77.4   | 84.4    | 74.0    |
# Prerequisites
We need two cuda environment to run this project, one for schema linking, another for sql generate.
## Schema Linking Environment
Create a virtual anaconda environment:
```
conda create -n schema_linking_env python=3.8.5
```
Active it and install the cuda version Pytorch:
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Install other required modules and tools:
```
pip install -r schema_linking_requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
python nltk_downloader.py
```
# SQL Generation Environment
Create a virtual anaconda environment:
```
conda create -n sql_generate_env python=3.8.5
```
Active it and install the cuda version Pytorch:
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Install other required modules and tools:
```
pip install -r sql_generate_requirements.txt
```
## Prepare data
Create directories to store prediction results and preprocessed data.
```commandline
mkdir data
mkdir data/preprocessed_data
mkdir data/predict_result
mkdir database
```
Download data from here [data](https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download).
```commandline
unzip spider.zip
mv spider/database/* database
mv spider data
```
# Inference
```commandline
python nltk_downloader.py
conda activate schema_linking_env
sh ./scripts/data_preprocessing.sh
conda activate sql_generate_env
sh ./scripts/inference.sh
```
The result of executing this command is a file located at `./data/predict_result/sqlgpt.sql`.
