# SQuAD-QA
Question answering on SQuAD 1.1 dataset

## Downloading Dataset
In order to download SQuAD 1.1 dataset, run the following commands:
```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```
This will download both training dataset (40 MB) and a development dataset (4 MB).

# Setting the environment
To setup the environment run the following commands:
```bash
conda create -n squad_qa_pytorch python=3.8
conda activate squad_qa_pytorch

conda install pytorch cudatoolkit=10.2 -c pytorch
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install gensim
pip install nltk
pip install transformers

# setup jupyter notebook support
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=squad_qa_pytorch
```
