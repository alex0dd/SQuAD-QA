print("Importing libraries...")
# External imports
import sys
import copy
import nltk
import numpy as np
import pandas as pd
import string
import torch
import json

from functools import partial
from nltk.tokenize import TreebankWordTokenizer, SpaceTokenizer
from transformers import AutoTokenizer
from typing import Tuple, List, Dict, Any, Union

# Project imports
from squad_data.parser import SquadFileParser
from squad_data.utils import build_mappers_and_dataframe_bert_eval
from evaluation.evaluate import evaluate_predictions
from evaluation.utils import build_evaluation_dict_bert
from utils import split_dataframe
from data_loading.utils import bert_padder_collate_fn_eval
from data_loading.qa_dataset import CustomQADatasetBERT_eval

# Pytorch model related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from timeit import default_timer as timer
from tqdm import tqdm
from transformers.optimization import AdamW

from models import possible_models_dict
from models.utils import SpanExtractor

USE_AMP = True

def bert_tokenizer_fn(question, paragraph, tokenizer, max_length=384, doc_stride=128):
    pad_on_right = tokenizer.padding_side == "right"
    # Process the sample
    tokenized_input_pair = tokenizer(
        question,
        paragraph,
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    return tokenized_input_pair

class ParametricBertModelQA(torch.nn.Module):

    def __init__(self, hidden_size, num_labels, config_dict, dropout_rate=0.3):
        super(ParametricBertModelQA, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.bert = transformers.AutoModel.from_pretrained(config_dict["model_url"])#(bert_config)
        self.bert_drop = torch.nn.Dropout(dropout_rate)
        self.qa_outputs = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.prepare_input_fn = config_dict["prepare_model_input_fn"]

    #@torch.cuda.amp.autocast() # goes OOM for whatever reason, don't use.
    def forward(self, inputs):
        # --- 1) Extract data from inputs dictionary and put it on right device
        curr_device = self.bert.device
        # --- 2) Run BERT backbone to produce final representation
        input_dict_for_bert = self.prepare_input_fn(inputs, curr_device)
        output = self.bert(**input_dict_for_bert)
        # --- 3) On top of the final representation, run a mapper to get scores for each position.
        sequence_output = output[0]   #(None, seq_len, hidden_size)
        # do dropout
        sequence_output = self.bert_drop(sequence_output)
        logits = self.qa_outputs(sequence_output) #(None, seq_len, hidden_size)*(hidden_size, 2)=(None, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)    #(None, seq_len, 1), (None, seq_len, 1)
        start_logits = start_logits.squeeze(-1)  #(None, seq_len)
        end_logits = end_logits.squeeze(-1)    #(None, seq_len)
        # --- 4) Prepare output tuple
        outputs = (start_logits, end_logits,) 
        return outputs

def main(path_to_json_file):
	# Choose model to use
	selected_model_name = "distilroberta"
	params_dict = possible_models_dict[selected_model_name]
	model_weights_filename = "trained_models/distilroberta_google_2_epochs.pt"
	# Load and parse the data
	print("Parsing the data...")
	parser = SquadFileParser(path_to_json_file)
	data = parser.parse_documents()
	#Prepare the tokenizer
	tokenizer = AutoTokenizer.from_pretrained(params_dict["tokenizer_url"])
	tokenizer_fn_preprocess = partial(bert_tokenizer_fn, tokenizer=tokenizer, max_length=params_dict["tokenizer_max_length"]-3)
	tokenizer_fn_train = partial(bert_tokenizer_fn, tokenizer=tokenizer, max_length=params_dict["tokenizer_max_length"])
	# Set the device
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# Define baseline model
	model = params_dict["span_model"](768, 2, params_dict, dropout_rate=params_dict["train_params"]["dropout_rate"]).to(device)
	scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
	# Load model from disk
	print("Loading Model weights...")
	if device == "cuda":
		model.load_state_dict(torch.load(model_weights_filename))
	else:
		model.load_state_dict(torch.load(model_weights_filename, map_location=torch.device('cpu'))) #cpu
	# Preprocess the input data
	print("Preprocessing data...")
	paragraphs_mapper, df = build_mappers_and_dataframe_bert_eval(tokenizer, tokenizer_fn_preprocess, data)
	# Prepare the data loader for the model
	dataset_QA = CustomQADatasetBERT_eval(tokenizer_fn_train, df, paragraphs_mapper)
	data_loader = torch.utils.data.DataLoader(dataset_QA, collate_fn=bert_padder_collate_fn_eval, batch_size=params_dict["train_params"]["batch_size_test"], shuffle=True)
	# Compute the predictions dictionary using the model
	print("Computing predictions...")
	pred_dict = build_evaluation_dict_bert(model, scaler, data_loader, paragraphs_mapper, tokenizer, device, show_progress=True)
	# Save the dictionary as a JSON file
	with open("predictions.txt", "w") as pred_file:
		pred_file.write(json.dumps(pred_dict))
	print("Predictions file created successfully")


if __name__ == '__main__':
	if len(sys.argv) > 1:
		main(sys.argv[1])
	else:
		print("Path to json file not found")
