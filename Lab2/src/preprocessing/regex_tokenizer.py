import re
from src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str):
        # B1: đưa text về lowercase
        text = text.lower()

        # B2: dùng regex để tách token
        tokens = re.findall(r"\w+|[^\w\s]", text)

        return tokens