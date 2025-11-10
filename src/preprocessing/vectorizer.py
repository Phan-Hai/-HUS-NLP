import numpy as np
from typing import List, Dict
from collections import Counter
import math

class Vectorizer:
    """Base class for text vectorizers"""
    
    def fit(self, texts: List[str]):
        """Learn vocabulary from texts"""
        raise NotImplementedError
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to vectors"""
        raise NotImplementedError
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)


class CountVectorizer(Vectorizer):
    """
    Convert text documents to a matrix of token counts.
    """
    
    def __init__(self, tokenizer, max_features: int = None, min_df: int = 1):
        """
        Initialize CountVectorizer.
        
        Args:
            tokenizer: Tokenizer instance to split text
            max_features: Maximum number of features (vocabulary size)
            min_df: Minimum document frequency for a term to be included
        """
        self.tokenizer = tokenizer
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_ = {}
        self.idf_ = None
    
    def fit(self, texts: List[str]):
        """
        Learn vocabulary from training texts.
        
        Args:
            texts: List of text documents
        """
        # Tokenize all texts
        tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
        
        # Count document frequency for each term
        doc_freq = Counter()
        for tokens in tokenized_texts:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Filter by min_df
        filtered_terms = {term: freq for term, freq in doc_freq.items() 
                         if freq >= self.min_df}
        
        # Sort by frequency and limit by max_features
        sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
        if self.max_features:
            sorted_terms = sorted_terms[:self.max_features]
        
        # Create vocabulary mapping
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to count matrix.
        
        Args:
            texts: List of text documents
            
        Returns:
            Count matrix of shape (n_samples, n_features)
        """
        if not self.vocabulary_:
            raise ValueError("Vocabulary not fitted. Call fit() first.")
        
        n_samples = len(texts)
        n_features = len(self.vocabulary_)
        
        # Initialize count matrix
        X = np.zeros((n_samples, n_features), dtype=np.float64)
        
        # Fill count matrix
        for i, text in enumerate(texts):
            tokens = self.tokenizer.tokenize(text)
            token_counts = Counter(tokens)
            
            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    j = self.vocabulary_[token]
                    X[i, j] = count
        
        return X


class TfidfVectorizer(CountVectorizer):
    """
    Convert text documents to TF-IDF feature matrix.
    TF-IDF = Term Frequency * Inverse Document Frequency
    """
    
    def fit(self, texts: List[str]):
        """
        Learn vocabulary and IDF from training texts.
        
        Args:
            texts: List of text documents
        """
        # First fit vocabulary using parent class
        super().fit(texts)
        
        # Calculate IDF
        n_samples = len(texts)
        tokenized_texts = [self.tokenizer.tokenize(text) for text in texts]
        
        # Count documents containing each term
        doc_freq = np.zeros(len(self.vocabulary_))
        for tokens in tokenized_texts:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    doc_freq[idx] += 1
        
        # Calculate IDF: log((N + 1) / (df + 1)) + 1
        self.idf_ = np.log((n_samples + 1) / (doc_freq + 1)) + 1
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF matrix.
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF matrix of shape (n_samples, n_features)
        """
        if self.idf_ is None:
            raise ValueError("IDF not calculated. Call fit() first.")
        
        # Get count matrix (TF)
        X = super().transform(texts)
        
        # Apply IDF weighting
        X = X * self.idf_
        
        # L2 normalization
        norms = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
        norms[norms == 0] = 1  # Avoid division by zero
        X = X / norms
        
        return X