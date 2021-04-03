# SQuAD-QA
Question answering on SQuAD 1.1 dataset

## Downloading Dataset
In order to download SQuAD 1.1 dataset, run the following commands:
```bash
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```
This will download both training dataset (40 MB) and a development dataset (4 MB).

## Setting the environment
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
pip install tqdm

# setup jupyter notebook support
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=squad_qa_pytorch
```

## Training and evaluation
Training of BERT models is supported via [Train_model_GPU_BERT.ipynb](https://github.com/alexpod1000/SQuAD-QA/blob/main/Train_model_GPU_BERT.ipynb) and [Train_model_TPU_BERT.ipynb](https://github.com/alexpod1000/SQuAD-QA/blob/main/Train_model_TPU_BERT.ipynb) notebooks.

The supported experiments can be found inside [this](https://github.com/alexpod1000/SQuAD-QA/blob/main/models/__init__.py) configurations file and include the following models: BERT, DistilBERT, ALBERT, DistilROBERTA.

## Results
The trained models obtain the following results on SQuAD-1.1 dev benchmark:
| Model                        | F1            | EM    |
| ---------------------------- |:-------------:| -----:|
| ALBERT                       | 86.97         | 78.39 |
| BERT                         | 81.51         | 71.82 |
| DistilBERT                   | 78.96         | 68.68 |
| DistilROBERTA                | 87.74         | 80.39 |
| DistilROBERTA + Linear Layer | **88.09**     | **80.93** |
