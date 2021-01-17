import torch
import numpy as np
from typing import Tuple, List


from nltk.tokenize import MWETokenizer, TreebankWordTokenizer

class AugmentedAnswerSpanning:
    
    def __init__(self):
        self.treebank_tokenizer = TreebankWordTokenizer()
        self.mwe_tokenizer = MWETokenizer(separator='')
        self.mwe_tokenizer.add_mwe(('<', 'ANS_START', '>'))
        self.mwe_tokenizer.add_mwe(('<', 'ANS_END', '>'))

        self.start_indicator = "<ANS_START>"
        self.end_indicator = "<ANS_END>"

    def __add_in_middle(self, string, pos, to_add):
        """
        Given a string, a position and a substring, 
        adds the substring at position index of the string.
        """
        return string[:pos] + to_add + string[pos:]

    def augment_string(self, string, answer_start_idx, answer_end_idx):
        """
        Given a string, adds the start and end indicators at right indexes
        """
        start_aug = self.__add_in_middle(string, answer_start_idx, self.start_indicator)
        end_aug = self.__add_in_middle(start_aug, len(self.start_indicator) + answer_end_idx, self.end_indicator)
        return end_aug

    def get_indexes_from_augmented_string(self, string):
        tokenized_aug = self.mwe_tokenizer.tokenize(self.treebank_tokenizer.tokenize(string))
        # get start of answer span index
        index_of_start_indicator = tokenized_aug.index(self.start_indicator)
        # remove index from string (now it will coincide with span start)
        tokenized_aug.pop(index_of_start_indicator)
        # same procedure for the end of span
        index_of_end_indicator = tokenized_aug.index(self.end_indicator)
        tokenized_aug.pop(index_of_end_indicator)
        index_of_end_indicator -= 1
        return tokenized_aug, (index_of_start_indicator, index_of_end_indicator)


class DataConverter:

    def __init__(self, embedding_model, paragraphs_spans_mapper):
        self.embedding_dict = {}
        self.embedding_dim = embedding_model.vector_size
        self.embedding_model = embedding_model
        self.paragraphs_spans_mapper = paragraphs_spans_mapper
        self.oovs_memory = {}
  
    def word_sequence_to_embedding(self, seq: List[str]) -> torch.Tensor:
        """
        Given a list of string words/tokens, returns their embedding
        according to the class's embedding model.
        
        Args:
            seq (List[str]): input sequence of words/tokens to embed.
        Returns:
            embeddings (torch.Tensor): torch tensor containing 
                the embeddings for each word of the input sequence.
        """
        
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

    def encode_answer(self, paragraph_id: int, answer_start: int, answer_text: str):
        """
        Returns:
            A torch.tensor with shape [2] where the first element is 
            the span_start and the second one is the span_stop
        """
        answer_end = answer_start + len(answer_text)
        answer_span = []
        for idx, span in enumerate(self.paragraphs_spans_mapper[paragraph_id]['spans']):
            if not (answer_end <= span[0] or answer_start >= span[1]):
                answer_span.append(idx)
        return torch.tensor([answer_span[0], answer_span[-1]]) 

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
    paragraph_id = [sample[3] for sample in sample_list]
    question_id = [sample[4] for sample in sample_list]
    paragraph_emb_padded = torch.nn.utils.rnn.pad_sequence(paragraph_emb, batch_first=True)
    question_emb_padded = torch.nn.utils.rnn.pad_sequence(question_emb, batch_first=True)
    #answer_emb_padded = torch.nn.utils.rnn.pad_sequence(out, batch_first=True)
    return {"paragraph_emb":paragraph_emb_padded,
            "question_emb":question_emb_padded,
            "y_gt":torch.stack(out),
            "paragraph_id":paragraph_id,
            "question_id":question_id}
