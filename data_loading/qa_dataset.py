import torch

class CustomQADatasetBERT(torch.utils.data.Dataset):
    """Custom text dataset for Huggingface BERT models."""

    def __init__(self, tokenizer_fn, df, paragraphs_mapper):
        super(CustomQADatasetBERT, self).__init__()
        self.input_list = df[["paragraph_id", "question_text", "question_id"]]
        self.output_list = df[["tokenizer_answer_start", "tokenizer_answer_end"]]
        self.paragraphs_mapper = paragraphs_mapper
        self.tokenizer_fn = tokenizer_fn

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        paragraph_id = self.input_list.iloc[idx]["paragraph_id"]
        question_id = self.input_list.iloc[idx]["question_id"]
        question_text = self.input_list.iloc[idx]["question_text"]
        tokenizer_answer_start = self.output_list.iloc[idx]["tokenizer_answer_start"]
        tokenizer_answer_end = self.output_list.iloc[idx]["tokenizer_answer_end"]

        paragraph_text = self.paragraphs_mapper[paragraph_id]
        tokenized_input_pair = self.tokenizer_fn(question_text, paragraph_text)
        
        #input_ids = torch.tensor(tokenized_input_pair["input_ids"], dtype=torch.long)
        #attention_mask = torch.tensor(tokenized_input_pair["attention_mask"], dtype=torch.long)
        input_ids = tokenized_input_pair["input_ids"]
        attention_mask = tokenized_input_pair["attention_mask"]

        out_span = torch.tensor([tokenizer_answer_start, tokenizer_answer_end])
        
        # NOTE: DistilBERT doesn’t have token_type_ids
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "out_span": out_span,
            "paragraph_id": paragraph_id,
            "question_id": question_id
        }

class CustomQADataset(torch.utils.data.Dataset):
    """Custom text dataset."""

    def __init__(self, data_converter, df, paragraphs_mapper):
        super(CustomQADataset, self).__init__()
        self.input_list = df[["paragraph_id", "question_id", "question_text"]]
        self.output_list = df[["answer_id", "answer_start", "answer_text"]]
        self.paragraphs_mapper = paragraphs_mapper
        self.data_converter = data_converter

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        paragraph_id = self.input_list.iloc[idx]["paragraph_id"]
        question_id = self.input_list.iloc[idx]["question_id"]
        question_text = self.input_list.iloc[idx]["question_text"]
        answer_start = self.output_list.iloc[idx]["answer_start"]
        answer_text = self.output_list.iloc[idx]["answer_text"]

        paragraph_text = self.paragraphs_mapper[paragraph_id]
        paragraph_emb = self.data_converter.word_sequence_to_embedding(paragraph_text)

        question_emb = self.data_converter.word_sequence_to_embedding(question_text)

        out = self.data_converter.encode_answer(paragraph_id, answer_start, answer_text)
        """
        # for each token, assign a class (1 -> nothing, 2 -> start/end of answer)
        out_emb_start = torch.ones(paragraph_emb.shape[0])
        out_emb_end = torch.ones(paragraph_emb.shape[0])
        out_emb_start[out[0]] = 2 # arbitrary class for start
        # end of answer class can be the same as start, 
        # since they will be classified by different heads
        out_emb_end[out[1]] = 2 # arbitrary class for end
        out = torch.stack([out_emb_start, out_emb_end], dim=-1)
        """
        
        return paragraph_emb, question_emb, out, paragraph_id, question_id
