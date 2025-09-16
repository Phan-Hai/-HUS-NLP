from src.core.interfaces import Vectorizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_ = {}

    def fit(self, corpus: list[str]):
        unique_tokens = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)
        sorted_tokens = sorted(unique_tokens)
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted_tokens)}

    def transform(self, documents: list[str]) -> list[list[int]]:
        vectors = []
        vocab_size = len(self.vocabulary_)
        for doc in documents:
            vector = [0] * vocab_size
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    vector[idx] += 1
            vectors.append(vector)
        return vectors
    
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        self.fit(corpus)
        return self.transform(corpus)