import torch

class CustomQADataset(torch.utils.data.Dataset):
    """Custom text dataset."""

    def __init__(self, data_converter, df, paragraphs_mapper, questions_mapper):
        self.input_list = df[["paragraph_id", "question_id"]]
        self.output_list = df[["answer_id", "answer_start", "answer_text"]]
        self.paragraphs_mapper = paragraphs_mapper
        self.questions_mapper = questions_mapper
        self.data_converter = data_converter

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        paragraph_id = self.input_list.iloc[idx]["paragraph_id"]
        question_id = self.input_list.iloc[idx]["question_id"]
        answer_start = self.output_list.iloc[idx]["answer_start"]
        answer_text = self.output_list.iloc[idx]["answer_text"]

        paragraph_text = self.paragraphs_mapper[paragraph_id]
        paragraph_emb = self.data_converter.word_sequence_to_embedding(paragraph_text)

        question_text = self.questions_mapper[question_id]
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
