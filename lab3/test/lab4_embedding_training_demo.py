import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

class SentenceStream:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield simple_preprocess(line)

data_path = r"D:\10. ky1nam4\NLP\data\UD_English-EWT\en_ewt-ud-train.txt"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)


sentences = SentenceStream(data_path)

print("Training Word2Vec model...")
model = Word2Vec(
    sentences=sentences,
    vector_size=100,     
    window=5,            
    min_count=3,         
    workers=8,           
    sg=1                 
)

print("Huấn luyện xong!")

model_path = os.path.join(results_dir, "word2vec_ewt.model")
model.save(model_path)
print("Done saving")

# Demo
print("Tìm các từ gần nghĩa với 'computer'")
if "computer" in model.wv:
    for w, s in model.wv.most_similar("computer", topn=5):
        print(f"{w:15s} {s:.4f}")
else:
    print("Từ 'computer' không có trong từ vựng mô hình.")


print("Ví dụ: king - man + woman ≈ ?")
try:
    result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
    print(f"Kết quả gần đúng: {result[0][0]} ({result[0][1]:.4f})")
except KeyError:
    print("Một trong các từ không tồn tại trong từ vựng mô hình.")
