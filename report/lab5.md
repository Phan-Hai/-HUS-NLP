# BÁO CÁO LAB 5: Xây dựng mô hình RNN cho bài toán POS Tagging

## 1. Mục tiêu
Trong bài thực hành này, chúng ta áp dụng kiến thức về mạng nơ-ron hồi quy (RNN) để xây dựng mô hình dự đoán nhãn Part-of-Speech (POS) cho từng token trong câu. Các bước chính:
- Tiền xử lý dữ liệu từ định dạng CoNLL-U.
- Xây dựng từ điển cho từ và nhãn.
- Tạo Dataset và DataLoader trong PyTorch.
- Xây dựng mô hình RNN đơn giản với các lớp `Embedding`, `RNN`, và `Linear`.
- Huấn luyện và đánh giá mô hình.

## 2. Bộ dữ liệu
- Sử dụng bộ dữ liệu **UD_English-EWT** ở định dạng CoNLL-U.
- Số lượng câu:
  - Train: **12,544 câu**
  - Dev: **2,001 câu**
- Ví dụ một câu sau khi xử lý: [('Al', 'PROPN'), ('-', 'PUNCT'), ('Zaman', 'PROPN'), (':', 'PUNCT'), ('American', 'ADJ')]
- Kích thước từ điển:
- Vocabulary: **19,675 từ**
- Tagset: **18 nhãn**
- Các nhãn: `<PAD>`, PROPN, PUNCT, ADJ, NOUN, VERB, DET, ADP, AUX, PRON, PART, SCONJ, NUM, ADV, CCONJ, X, INTJ, SYM.

## 3. Mô hình
- Kiến trúc: SimpleRNNForTokenClassification(
(embedding): Embedding(19675, 100, padding_idx=0)
(rnn): RNN(100, 128, batch_first=True)
(fc): Linear(in_features=128, out_features=18, bias=True)
)

## 4. Huấn luyện
- Số epoch: **10**
- Optimizer: Adam
- Loss: CrossEntropyLoss (ignore_index cho padding)
- Kết quả huấn luyện:

| Epoch | Train Loss | Train Acc | Dev Acc |
|-------|-----------|-----------|---------|
| 1     | 1.1228    | 0.7700    | 0.7477 |
| 5     | 0.3058    | 0.9208    | 0.8548 |
| 10    | 0.1275    | 0.9684    | 0.8744 |

- **Best Dev Accuracy:** **0.8744**

## 5. Dự đoán câu mới
- **Câu:** "I love NLP"  
**Dự đoán:** `[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'NOUN')]`

- **Câu:** "The cat sits on the mat"  
**Dự đoán:** `[('The', 'DET'), ('cat', 'NOUN'), ('sits', 'NOUN'), ('on', 'ADP'), ('the', 'DET'), ('mat', 'NOUN')]`

- **Câu:** "She is reading a book"  
**Dự đoán:** `[('She', 'PRON'), ('is', 'AUX'), ('reading', 'VERB'), ('a', 'DET'), ('book', 'NOUN')]`