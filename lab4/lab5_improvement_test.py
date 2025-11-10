import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.tokenizer import RegexTokenizer
from src.preprocessing.vectorizer import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict
import pandas as pd


class ImprovedTextClassifier:
    """Improved Text Classifier with different model options."""
    
    def __init__(self, vectorizer, model_type='logistic'):
        self.vectorizer = vectorizer
        self.model_type = model_type
        self._model = None
    
    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        
        if self.model_type == 'naive_bayes':
            from sklearn.naive_bayes import MultinomialNB
            self._model = MultinomialNB(alpha=1.0)
        else:
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(
                solver='liblinear', 
                random_state=42,
                C=1.0,
                max_iter=200
            )
        
        self._model.fit(X, labels)
        return self
    
    def predict(self, texts: List[str]) -> List[int]:
        if self._model is None:
            raise ValueError("Model not trained.")
        
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

def load_sentiments_data(file_path):
    """
    Load sentiment data from CSV file.
    
    Args:
        file_path: Path to sentiments.csv
        
    Returns:
        texts: List of text strings
        labels: List of labels (0 or 1)
    """
    print(f"Loading data from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"✗ ERROR: File not found at {file_path}")
        return None, None
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Check required columns
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            print("✗ ERROR: Required columns 'text' and 'sentiment' not found")
            print(f"  Available columns: {df.columns.tolist()}")
            return None, None
        
        # Remove rows with missing values
        initial_count = len(df)
        df = df.dropna(subset=['text', 'sentiment'])
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"  Removed {initial_count - final_count} rows with missing values")
        
        # Get texts
        texts = df['text'].tolist()
        
        # Convert sentiment to 0/1 labels
        sentiments = df['sentiment'].tolist()
        
        # Check if sentiments are -1/1 or 0/1
        unique_sentiments = set(sentiments)
        print(f"\nUnique sentiment values: {unique_sentiments}")
        
        if -1 in unique_sentiments:
            # Convert -1/1 to 0/1
            labels = [(s + 1) // 2 for s in sentiments]
            print("Converted sentiment from {-1, 1} to {0, 1}")
        else:
            # Assume already 0/1
            labels = [int(s) for s in sentiments]
            print("Using sentiment values as-is")
        
        # Print distribution
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(texts)}")
        print(f"  Positive (1): {positive_count} ({positive_count/len(labels)*100:.1f}%)")
        print(f"  Negative (0): {negative_count} ({negative_count/len(labels)*100:.1f}%)")
        
        # Show sample data
        print(f"\nSample data:")
        for i in range(min(3, len(texts))):
            sentiment_label = "Positive" if labels[i] == 1 else "Negative"
            print(f"  [{sentiment_label}] {texts[i][:60]}...")
        
        return texts, labels
        
    except Exception as e:
        print(f"✗ ERROR reading CSV: {e}")
        return None, None


def main():
    print("=" * 70)
    print("Lab 5: Model Improvement Experiments")
    print("=" * 70)
    data_path = r"D:\10. ky1nam4\NLP\data\sentiments.csv"
    
    texts, labels = load_sentiments_data(data_path)
    
    if texts is None or labels is None:
        print("\n✗ Failed to load data. Exiting...")
        return
    
    if len(texts) < 10:
        print(f"\n✗ ERROR: Too few samples ({len(texts)}). Need at least 10 samples.")
        return
    
    print("\n" + "-" * 70)
    print("Splitting Data")
    print("-" * 70)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, 
            test_size=0.25, 
            random_state=42, 
            stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Train positive ratio: {sum(y_train)/len(y_train)*100:.1f}%")
        print(f"Test positive ratio: {sum(y_test)/len(y_test)*100:.1f}%")
        
    except Exception as e:
        print(f"✗ ERROR during split: {e}")
        print("Using simple split without stratification...")
        split_idx = int(len(texts) * 0.75)
        X_train, X_test = texts[:split_idx], texts[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]
    
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Experiment 1: Baseline
    print("\n" + "=" * 70)
    print("Experiment 1: Baseline (LogReg + TF-IDF, max_feat=100)")
    print("=" * 70)
    
    tokenizer1 = RegexTokenizer(pattern=r'\b\w+\b')
    vectorizer1 = TfidfVectorizer(tokenizer1, max_features=100, min_df=1)
    classifier1 = ImprovedTextClassifier(vectorizer1, model_type='logistic')
    
    classifier1.fit(X_train, y_train)
    y_pred1 = classifier1.predict(X_test)
    metrics1 = classifier1.evaluate(y_test, y_pred1)
    
    print(f"Accuracy: {metrics1['accuracy']:.4f}, F1: {metrics1['f1_score']:.4f}")
    
    # Experiment 2: Naive Bayes
    print("\n" + "=" * 70)
    print("Experiment 2: Naive Bayes + TF-IDF")
    print("=" * 70)
    
    tokenizer2 = RegexTokenizer(pattern=r'\b\w+\b')
    vectorizer2 = TfidfVectorizer(tokenizer2, max_features=100, min_df=1)
    classifier2 = ImprovedTextClassifier(vectorizer2, model_type='naive_bayes')
    
    classifier2.fit(X_train, y_train)
    y_pred2 = classifier2.predict(X_test)
    metrics2 = classifier2.evaluate(y_test, y_pred2)
    
    print(f"Accuracy: {metrics2['accuracy']:.4f}, F1: {metrics2['f1_score']:.4f}")
    
    # Experiment 3: More Features
    print("\n" + "=" * 70)
    print("Experiment 3: LogReg + More Features (max_feat=200)")
    print("=" * 70)
    
    tokenizer3 = RegexTokenizer(pattern=r'\b\w+\b')
    vectorizer3 = TfidfVectorizer(tokenizer3, max_features=200, min_df=1)
    classifier3 = ImprovedTextClassifier(vectorizer3, model_type='logistic')
    
    classifier3.fit(X_train, y_train)
    y_pred3 = classifier3.predict(X_test)
    metrics3 = classifier3.evaluate(y_test, y_pred3)
    
    print(f"Accuracy: {metrics3['accuracy']:.4f}, F1: {metrics3['f1_score']:.4f}")
    
    # Experiment 4: Filtered Vocabulary
    print("\n" + "=" * 70)
    print("Experiment 4: LogReg + Filter Rare Words (min_df=2)")
    print("=" * 70)
    
    tokenizer4 = RegexTokenizer(pattern=r'\b\w+\b')
    vectorizer4 = TfidfVectorizer(tokenizer4, max_features=150, min_df=2)
    classifier4 = ImprovedTextClassifier(vectorizer4, model_type='logistic')
    
    classifier4.fit(X_train, y_train)
    y_pred4 = classifier4.predict(X_test)
    metrics4 = classifier4.evaluate(y_test, y_pred4)
    
    print(f"Accuracy: {metrics4['accuracy']:.4f}, F1: {metrics4['f1_score']:.4f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    experiments = [
        ("Baseline (LogReg, max_feat=100)", metrics1),
        ("Naive Bayes", metrics2),
        ("More Features (max_feat=200)", metrics3),
        ("Filter Rare (min_df=2)", metrics4)
    ]
    
    print(f"\n{'Experiment':<35} {'Accuracy':<12} {'F1-Score'}")
    print("-" * 70)
    
    for name, metrics in experiments:
        print(f"{name:<35} {metrics['accuracy']:.4f} ({metrics['accuracy']*100:5.2f}%)  {metrics['f1_score']:.4f}")
    
    best_idx = max(range(len(experiments)), key=lambda i: experiments[i][1]['f1_score'])
    best_name, best_metrics = experiments[best_idx]
    
    print(f"\nBEST: {best_name}")
    print(f"   F1: {best_metrics['f1_score']:.4f}, Acc: {best_metrics['accuracy']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()