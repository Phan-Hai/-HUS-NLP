from src.representations.word_embedder import WordEmbedder

# Khởi tạo model
embedder = WordEmbedder("glove-wiki-gigaword-50")

# Lấy vector của một từ
print("\nVector for 'king':")
print(embedder.get_vector("king"))

# Tính similarity
print("\nSimilarity(king, queen):", embedder.get_similarity("king", "queen"))
print("Similarity(king, man):", embedder.get_similarity("king", "man"))

# Tìm từ gần nghĩa
print("\nMost similar to 'computer':")
for w, s in embedder.get_most_similar("computer"):
    print(f"{w:15s} {s:.4f}")

# Biểu diễn câu thành vector văn bản
sentence = "The queen rules the country."
doc_vector = embedder.embed_document(sentence)
print(f"\nDocument embedding for: \"{sentence}\"")
print(doc_vector)
print(f"\nVector dimension: {len(doc_vector)}")