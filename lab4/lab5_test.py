import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.tokenizer import RegexTokenizer
from src.preprocessing.vectorizer import TfidfVectorizer
from src.models.text_classifier import TextClassifier
from sklearn.model_selection import train_test_split

def main():
    print("=" * 60)
    print("Lab 5: Text Classification - Basic Test")
    print("=" * 60)
    
    # Dataset
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad.",
        "Amazing cinematography and brilliant plot!",
        "Worst movie I've ever seen.",
        "Beautiful story, wonderful acting.",
        "Terrible waste of money.",
        "Absolutely loved every minute of it!",
        "Boring and predictable.",
        "Best film of the year!",
        "Not worth watching at all.",
        "Incredible performances by all actors.",
        "Disappointed and frustrated."
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    print(f"\nTotal samples: {len(texts)}")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Initialize components
    print("\n" + "-" * 60)
    print("Step 1: Initialize Tokenizer and Vectorizer")
    print("-" * 60)
    tokenizer = RegexTokenizer(pattern=r'\b\w+\b')
    vectorizer = TfidfVectorizer(tokenizer, max_features=100, min_df=1)
    print("✓ Tokenizer and Vectorizer initialized")
    
    # Initialize classifier
    print("\n" + "-" * 60)
    print("Step 2: Initialize and Train Classifier")
    print("-" * 60)
    classifier = TextClassifier(vectorizer)
    
    # Train the model
    print("Training Logistic Regression model...")
    classifier.fit(X_train, y_train)
    print("✓ Model trained successfully")
    
    # Make predictions
    print("\n" + "-" * 60)
    print("Step 3: Make Predictions")
    print("-" * 60)
    y_pred = classifier.predict(X_test)
    print("✓ Predictions completed")
    
    # Display predictions
    print("\nTest Predictions:")
    for i, (text, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):
        status = "✓" if true_label == pred_label else "✗"
        sentiment_true = "Positive" if true_label == 1 else "Negative"
        sentiment_pred = "Positive" if pred_label == 1 else "Negative"
        print(f"{status} Text: {text[:50]}...")
        print(f"  True: {sentiment_true}, Predicted: {sentiment_pred}\n")
    
    # Evaluate
    print("-" * 60)
    print("Step 4: Evaluate Model Performance")
    print("-" * 60)
    metrics = classifier.evaluate(y_test, y_pred)
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()