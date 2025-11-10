from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextClassifier:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self._model = None
    
    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        
        self._model = LogisticRegression(solver='liblinear', random_state=42)
        self._model.fit(X, labels)
        
        return self
    
    def predict(self, texts: List[str]) -> List[int]:
        if self._model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X = self.vectorizer.transform(texts)
        predictions = self._model.predict(X)
        
        return predictions.tolist()
    
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        return metrics


