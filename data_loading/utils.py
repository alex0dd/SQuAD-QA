import torch
import numpy as np

class DataConverter:

    def __init__(self, embedding_model):
        self.embedding_dict = {}
        self.embedding_dim = embedding_model.vector_size
        self.embedding_model = embedding_model
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



def padder_collate_fn(sample_list):
    paragraph_emb = [sample[0] for sample in sample_list]
    question_emb = [sample[1] for sample in sample_list]
    out = [sample[2] for sample in sample_list]
    paragraph_emb_padded = torch.nn.utils.rnn.pad_sequence(paragraph_emb)
    question_emb_padded = torch.nn.utils.rnn.pad_sequence(question_emb)
    return paragraph_emb_padded, question_emb_padded, out