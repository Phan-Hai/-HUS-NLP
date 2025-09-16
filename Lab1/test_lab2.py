from src.representations.count_vectorizer import CountVectorizer
from src.preprocessing.regex_tokenizer import RegexTokenizer


def main():
    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
    vectors = vectorizer.fit_transform(corpus)
    print("Vocabulary:", vectorizer.vocabulary_)
    print("Document-term matrix:")
    for vector in vectors:
        print(vector)

if __name__ == "__main__":
    main()