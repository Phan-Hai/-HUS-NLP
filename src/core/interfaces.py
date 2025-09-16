# src/core/interfaces.py
from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tách chuỗi text thành danh sách token"""
        pass

class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]):
        pass

    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        pass

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        self.fit(corpus)
        return self.transform(corpus)