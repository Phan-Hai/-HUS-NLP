import logging
from src.representations.count_vectorizer import CountVectorizer
from src.preprocessing.regex_tokenizer import RegexTokenizer

logging.basicConfig(
    filename="lab2/run.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def test_count_vectorizer():
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    dtm = vectorizer.fit_transform(corpus)

    logging.info("Vocabulary: %s", vectorizer.vocabulary_)
    logging.info("Document-Term Matrix:")
    for row in dtm:
        logging.info("%s", row)

if __name__ == "__main__":
    test_count_vectorizer()
    print("Output đã ghi vào run.log")
