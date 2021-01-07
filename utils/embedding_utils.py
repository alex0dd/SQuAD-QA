import os
import gensim
import gensim.downloader as gloader
from gensim.models import KeyedVectors

class EmbeddingDownloader:

    def __init__(self, directory: str, filename: str, model_name: str) -> None:
        self.directory = directory
        self.filename = filename
        self.model_name = model_name
        self.model_path = os.path.join(os.getcwd(), self.directory, self.filename)

    def __download_embedding_model(self, download_path: str) -> KeyedVectors:
        """
        Downloads a pre-trained word embedding model via gensim library.
        :return
            - pre-trained word embedding model (gensim KeyedVectors object)
        """
        # Check download
        try:
            emb_model = gloader.load(download_path)
        except ValueError as e:
            print("Invalid embedding model name!")
            raise e
        return emb_model

    def load(self) -> KeyedVectors:
        embedding_model = None
        if not os.path.exists(self.model_path):
            print("Downloading embedding into {}".format(self.model_path))
            # NOTE: this might take around 5min (1GB to download and unpack)
            embedding_model = self.__download_embedding_model(self.model_name)
            os.makedirs(self.directory)
            embedding_model.save(self.model_path)
        else:
            print("Loading pre-downloaded embeddings from {}".format(self.model_path))
            embedding_model = KeyedVectors.load(self.model_path)
        if embedding_model is not None:
            print("End!")
            print("Embedding dimension: {}".format(embedding_model.vector_size))
        else:
            print("Embedding model was not loaded.")
        return embedding_model

