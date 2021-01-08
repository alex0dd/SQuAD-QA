import torch
import numpy as np
from typing import Tuple

class DataConverter:

    def __init__(self, embedding_model, paragraphs_spans_mapper):
        self.embedding_dict = {}
        self.embedding_dim = embedding_model.vector_size
        self.embedding_model = embedding_model
        self.paragraphs_spans_mapper = paragraphs_spans_mapper
        self.oovs_memory = {}
  
    def word_sequence_to_embedding(self, seq):
        embeddings = []
        for w in seq:
            if w in self.embedding_model:
                # recover from gensim
                embedding_w = self.embedding_model.get_vector(w)
                embeddings.append(embedding_w)
            else:
                if w not in self.oovs_memory:
                    # assign random 
                    emb = np.random.randn(self.embedding_dim)
                    # store the random embedding for the next time
                    self.oovs_memory[w] = emb
                else:
                    emb = self.oovs_memory[w]
                embeddings.append(emb)
        return torch.tensor(embeddings, dtype=torch.float32)

    def encode_answer(self, paragraph_id: int, answer_start: int, answer_text: str) -> Tuple[int,int]:
        answer_end = answer_start + len(answer_text)
        answer_span = []
        for idx, span in enumerate(self.paragraphs_spans_mapper[paragraph_id]['spans']):
            if not (answer_end <= span[0] or answer_start >= span[1]):
                answer_span.append(idx)
        return answer_span[0], answer_span[-1]

    def decode_answer(self, paragraph_id: int, answer_encoded: Tuple[int,int]) -> str:
        text = self.paragraphs_spans_mapper[paragraph_id]['text']
        answer_start_index = self.paragraphs_spans_mapper[paragraph_id]['spans'][answer_encoded[0]] # (position, position + len(token))
        answer_stop_index = self.paragraphs_spans_mapper[paragraph_id]['spans'][answer_encoded[1]] # (position, position + len(token))

        if answer_start_index[0] >= len(text) or answer_start_index[1] > len(text):
            print("answer_start_index exceeds the len of the paragraph text")
            raise Exception()
        if answer_stop_index[0] > len(text) or answer_stop_index[1] > len(text):
            print("answer_stop_index exceeds the len of the paragraph text")
            raise Exception()

        return text[answer_start_index[0]:answer_stop_index[1]]

def padder_collate_fn(sample_list):
    paragraph_emb = [sample[0] for sample in sample_list]
    question_emb = [sample[1] for sample in sample_list]
    out = [sample[2] for sample in sample_list]
    paragraph_emb_padded = torch.nn.utils.rnn.pad_sequence(paragraph_emb)
    question_emb_padded = torch.nn.utils.rnn.pad_sequence(question_emb)
    return paragraph_emb_padded, question_emb_padded, out
