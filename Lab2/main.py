# main.py

from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

if __name__ == "__main__":
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    for s in sentences:
        print(f"\nOriginal: {s}")
        print(f"SimpleTokenizer: {simple_tokenizer.tokenize(s)}")
        print(f"RegexTokenizer:   {regex_tokenizer.tokenize(s)}")

    # Part 2: Test với UD_English-EWT dataset
    dataset_path = r"D:\10. ky1nam4\NLP\data\UD_English-EWT\en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    sample_text = raw_text[:500]

    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample (100 chars): {sample_text[:100]}...\n")

    simple_tokens = simple_tokenizer.tokenize(sample_text)
    regex_tokens = regex_tokenizer.tokenize(sample_text)

    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")
