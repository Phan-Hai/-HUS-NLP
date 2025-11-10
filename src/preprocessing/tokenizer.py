import re
from typing import List

class RegexTokenizer:
    """
    A simple tokenizer that uses regular expressions to split text into tokens.
    """
    
    def __init__(self, pattern: str = r'\b\w+\b'):
        """
        Initialize the tokenizer with a regex pattern.
        
        Args:
            pattern: Regular expression pattern for tokenization
        """
        self.pattern = pattern
        self.regex = re.compile(pattern, re.IGNORECASE)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens (words)
        """
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase and find all matches
        tokens = self.regex.findall(text.lower())
        return tokens
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize a batch of text strings.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of token lists
        """
        return [self.tokenize(text) for text in texts]