import gensim.downloader as api
import numpy as np
from src.preprocessing.regex_tokenizer import RegexTokenizer

class WordEmbedder:
    def __init__(self, model_name: str):
        try:
            print(f"Đang tải mô hình '{model_name}'...")
            self.model = api.load(model_name)
            print(f"Mô hình '{model_name}' đã được tải.")
        except ValueError:
            print(f"Mô hình '{model_name}' không tồn tại trong Gensim API.")
            self.model = None

    def get_vector(self, word: str):
        if word in self.model:
            return self.model[word]
        else:
            print(f"Từ '{word}' không có trong mô hình.")
            return None

    def get_similarity(self, word1: str, word2: str):
        if word1 in self.model and word2 in self.model:
            return self.model.similarity(word1, word2)
        else:
            print("Một trong hai từ không có trong mô hình.")
            return None

    def get_most_similar(self, word: str, top_n: int = 10):
        if word in self.model:
            return self.model.most_similar(word, topn=top_n)
        else:
            print(f"Từ '{word}' không có trong mô hình.")
            return None
    
    def embed_document(self, document: str):
        tokenizer = RegexTokenizer()
        tokens = tokenizer.tokenize(document)
        vectors = [self.model[token] for token in tokens if token in self.model]

        if not vectors:
            return np.zeros(self.dim)
        else:
            return np.mean(vectors, axis=0)