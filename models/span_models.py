import torch
import transformers

class ParametricBertModelQA(torch.nn.Module):

    def __init__(self, hidden_size, num_labels, config_dict, dropout_rate=0.3):
        super(ParametricBertModelQA, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.bert = transformers.AutoModel.from_pretrained(config_dict["model_url"])#(bert_config)
        self.bert_drop = torch.nn.Dropout(dropout_rate)
        self.qa_outputs = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.prepare_input_fn = config_dict["prepare_model_input_fn"]

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
    
    
class ExtraParametricBertModelQA(torch.nn.Module):

    def __init__(self, hidden_size, num_labels, config_dict, dropout_rate=0.3):
        super(ExtraParametricBertModelQA, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.bert = transformers.AutoModel.from_pretrained(config_dict["model_url"])#(bert_config)
        self.bert_drop = torch.nn.Dropout(dropout_rate)
        self.extra_linear = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.extra_linear_tanh = torch.nn.Tanh()
        self.qa_outputs = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.prepare_input_fn = config_dict["prepare_model_input_fn"]

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
        sequence_output = self.extra_linear(sequence_output)
        sequence_output = self.extra_linear_tanh(sequence_output)
        logits = self.qa_outputs(sequence_output) #(None, seq_len, hidden_size)*(hidden_size, 2)=(None, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)    #(None, seq_len, 1), (None, seq_len, 1)
        start_logits = start_logits.squeeze(-1)  #(None, seq_len)
        end_logits = end_logits.squeeze(-1)    #(None, seq_len)
        # --- 4) Prepare output tuple
        outputs = (start_logits, end_logits,) 
        return outputs