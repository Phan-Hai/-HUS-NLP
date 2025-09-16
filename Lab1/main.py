import logging
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
    filename="run.log",
    filemode="w"
)


if __name__ == "__main__":
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    # Part 1: Test với các câu mẫu
    logging.info("--- Tokenizing Sample Sentences ---")
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    for s in sentences:
        logging.info(f"Original: {s}")
        logging.info(f"SimpleTokenizer: {simple_tokenizer.tokenize(s)}")
        logging.info(f"RegexTokenizer:   {regex_tokenizer.tokenize(s)}")

    # Part 2: Test với UD_English-EWT dataset
    logging.info("--- Loading and Tokenizing from UD_English-EWT Dataset ---")
    dataset_path = r"D:\10. ky1nam4\NLP\data\UD_English-EWT\en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    logging.info(f"Đang load dataset từ {dataset_path}...")
    sample_text = raw_text[:500]

    logging.info("Tokenizing sample text từ UD_English-EWT...")
    logging.info(f"Original Sample (100 chars): {sample_text[:100]}...")

    simple_tokens = simple_tokenizer.tokenize(sample_text)
    regex_tokens = regex_tokenizer.tokenize(sample_text)

    logging.info(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
    logging.info(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")
