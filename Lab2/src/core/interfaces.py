# src/core/interfaces.py
from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tách chuỗi text thành danh sách token"""
        pass
