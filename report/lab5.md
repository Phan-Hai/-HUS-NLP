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

# Báo Cáo Lab 5: Phân loại Văn bản với RNN/LSTM

## 1. Tổng Quan Kết Quả Định Lượng

| Pipeline | F1-score (Macro) | Test Loss | Test Accuracy |
|----------|------------------|-----------|---------------|
| TF-IDF + Logistic Regression | **0.83** | N/A | **0.83** |
| Word2Vec (Avg) + Dense | 0.41 | 1.94 | 0.46 |
| Embedding (Pre-trained) + LSTM | 0.0005 | 4.12 | 0.02 |
| Embedding (Scratch) + LSTM | 0.0005 | 4.12 | 0.02 |

### Nhận Xét Chung về Kết Quả

Kết quả thí nghiệm cho thấy một bức tranh **ngược lại hoàn toàn** với kỳ vọng lý thuyết:
- **TF-IDF + LR đạt hiệu suất tốt nhất** (F1: 0.83)
- **Các mô hình LSTM hoàn toàn thất bại** (F1 ≈ 0, chỉ dự đoán 1-2 lớp)
- Word2Vec + Dense đạt kết quả trung bình (F1: 0.41)

---

## 2. Phân Tích Định Tính - Dự Đoán trên Câu Khó

### **Câu 1: "can you remind me to NOT call my mom"**
**Nhãn thực tế mong đợi:** `reminder_create`

| Mô hình | Dự đoán | Đánh giá |
|---------|---------|----------|
| TF-IDF + LR | `calendar_set` | Sai nhưng gần (cùng về quản lý thời gian) |
| Word2Vec + Dense | `social_post` | Sai hoàn toàn |
| LSTM (Pre-trained) | `general_dontcare` | Sai - mô hình bị collapse |
| LSTM (Scratch) | `play_audiobook` | Sai - mô hình bị collapse |

**Phân tích chi tiết:**
- Câu này có **phụ thuộc xa quan trọng**: từ "NOT" ở giữa câu phủ định hành động "call"
- **Ngữ cảnh phức tạp**: "remind to NOT do X" khác hoàn toàn với "remind to do X"
- **Kỳ vọng:** LSTM với khả năng xử lý chuỗi nên hiểu được mối quan hệ này
- **Thực tế:** LSTM thất bại hoàn toàn, không học được pattern nào
- **TF-IDF tốt hơn:** Dù không hiểu phủ định, nhưng các từ "remind", "me" đủ mạnh để phân loại vào nhóm task management

---

### **Câu 2: "is it going to be sunny OR rainy tomorrow"**
**Nhãn thực tế mong đợi:** `weather_query`

| Mô hình | Dự đoán | Đánh giá |
|---------|---------|----------|
| TF-IDF + LR | `weather_query` |**ĐÚNG** |
| Word2Vec + Dense | `weather_query` |**ĐÚNG** |
| LSTM (Pre-trained) | `general_dontcare` |Sai - mô hình bị collapse |
| LSTM (Scratch) | `play_audiobook` |Sai - mô hình bị collapse |

**Phân tích chi tiết:**
- Câu này có **từ khóa rõ ràng**: "sunny", "rainy", "tomorrow", "weather"
- **Cấu trúc OR:** "sunny OR rainy" cần hiểu logic chọn lựa
- **Kỳ vọng:** Cả bag-of-words và LSTM đều nên đoán đúng
- **Thực tế:** Chỉ TF-IDF và Word2Vec thành công
- **Nguyên nhân thành công của baseline:** Các từ domain-specific ("sunny", "rainy") là strong signals

---

### **Câu 3: "find a flight from new york to london BUT NOT through paris"**
**Nhãn thực tế mong đợi:** `transport_query` hoặc `flight_search`

| Mô hình | Dự đoán | Đánh giá |
|---------|---------|----------|
| TF-IDF + LR | `transport_query` | **ĐÚNG** hoặc gần đúng |
| Word2Vec + Dense | `email_sendemail` | Sai hoàn toàn |
| LSTM (Pre-trained) | `general_dontcare` | Sai - mô hình bị collapse |
| LSTM (Scratch) | `play_audiobook` | Sai - mô hình bị collapse |

**Phân tích chi tiết:**
- Đây là **câu phức tạp nhất** với:
  - Phụ thuộc xa: "BUT NOT through paris" phủ định điều kiện ở cuối
  - Ngữ cảnh đa thành phần: origin, destination, constraint
  - Yêu cầu hiểu logic điều kiện
- **Kỳ vọng:** LSTM với khả năng xử lý chuỗi dài nên vượt trội
- **Thực tế:** LSTM thất bại thảm hại
- **TF-IDF thành công:** Từ "flight", "new york", "london" đủ mạnh để phân loại đúng
- **Word2Vec sai nghiêm trọng:** Có thể do vector trung bình làm mất đi cấu trúc câu

---

## 3. Phân Tích Nguyên Nhân LSTM Thất Bại

### 3.1. Hiện Tượng "Model Collapse"

Cả hai mô hình LSTM đều cho thấy dấu hiệu **model collapse**:
- **Accuracy cực thấp** (~1.8%): Gần như random guessing (1/64 classes ≈ 1.56%)
- **Dự đoán chỉ 1-2 lớp:** "general_dontcare" và "play_audiobook" xuất hiện liên tục
- **Loss cực cao** (4.12): Mô hình không học được gì

### 3.2. Nguyên Nhân Có Thể

#### **A. Vấn đề về Dữ liệu**
1. **Kích thước tập train quá nhỏ** 
   - LSTM cần nhiều dữ liệu hơn nhiều so với phương pháp truyền thống
   - TF-IDF + LR có thể học tốt với ít dữ liệu hơn

2. **Mất cân bằng lớp nghiêm trọng**
   - Một số lớp có quá ít mẫu
   - LSTM có thể học bias về các lớp dominant

3. **Chất lượng tokenization**
   - Có thể vocab_size=5000 quá nhỏ
   - OOV (Out-of-Vocabulary) words không được xử lý tốt

#### **B. Vấn đề về Kiến trúc & Huấn luyện**

1. **Embedding Matrix không hiệu quả**
   ```
   Từ tìm thấy trong Word2Vec: X/Y (tỷ lệ thấp?)
   ```
   - Nhiều từ trong test set không có trong Word2Vec
   - Embedding pre-trained không trainable → không thể adapt

2. **Hyperparameters không phù hợp**
   - LSTM 128 units có thể quá lớn cho dataset nhỏ
   - Dropout 0.2 có thể chưa đủ
   - Batch size 32 có thể không tối ưu
   - Learning rate mặc định có thể không phù hợp

3. **Vanishing Gradient vẫn xảy ra**
   - Dù LSTM được thiết kế để giải quyết vấn đề này
   - Với max_len=50, vẫn có thể gặp khó khăn với phụ thuộc xa

4. **Khởi tạo trọng số không tốt**
   - Embedding từ Word2Vec có thể không phù hợp với domain
   - Khởi tạo ngẫu nhiên (scratch) càng tệ hơn với data ít

# Báo Cáo Lab 5: Xây dựng Mô hình RNN cho Bài toán Nhận dạng Thực thể Tên (NER)

* **Mục tiêu:** Xây dựng mô hình Mạng Nơ-ron Hồi quy (GRU) để phân loại thực thể tên trên từng token, sử dụng bộ dữ liệu CoNLL 2003.

---

## 1. Quá trình Thực hiện (Tasks 1-5)

### Task 1: Tải và Tiền xử lý Dữ liệu

1.  [cite_start]**Tải Dữ liệu:** Sử dụng thư viện `datasets` để tải bộ dữ liệu **CoNLL 2003**[cite: 13, 14].
    * **Khắc phục Lỗi API:** Do lỗi `AttributeError` khi truy cập thuộc tính `.names` của metadata, danh sách tên nhãn NER đã được **định nghĩa thủ công** theo chuẩn IOB2 của CoNLL 2003:
        ```python
        tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        ```
2.  **Xây dựng Từ điển (Vocabulary):**
    * **Kích thước Từ điển Từ (VOCAB_SIZE):** 23625
    * **Kích thước Từ điển Nhãn (OUTPUT_SIZE):** 9

---

### Task 2: Tạo PyTorch Dataset và DataLoader

1.  [cite_start]**Lớp `NERDataset`:** Xử lý việc ánh xạ từ và nhãn sang các chỉ số số nguyên (`token_indices`, `tag_indices`)[cite: 44, 47, 48].
2.  [cite_start]**Hàm `collate_fn`:** Thực hiện đệm (padding) động cho các câu và nhãn trong mỗi batch bằng `torch.nn.utils.rnn.pad_sequence`[cite: 52].
    * [cite_start]**Padding cho nhãn:** Sử dụng chỉ số đặc biệt **`PAD_TAG_IDX = -100`** để các token đệm bị bỏ qua trong tính toán hàm lỗi[cite: 53].

---

### Task 3: Xây dựng Mô hình RNN

[cite_start]Mô hình **`SimpleRNNForTokenClassification`** được xây dựng với kiến trúc 3 lớp[cite: 56]:
1.  **`nn.Embedding`**: Lớp embedding từ.
2.  [cite_start]**`nn.GRU`**: Sử dụng GRU (thay cho RNN cơ bản) để xử lý chuỗi và trích xuất ngữ cảnh[cite: 58].
3.  **`nn.Linear`**: Ánh xạ output của RNN sang không gian nhãn.

---

### Task 4: Huấn luyện Mô hình

1.  [cite_start]**Loss Function:** `nn.CrossEntropyLoss` được sử dụng[cite: 65, 66].
2.  [cite_start]**Thiết lập `ignore_index`:** Tham số **`ignore_index`** được đặt bằng **`-100`** để đảm bảo hàm lỗi bỏ qua các vị trí padding khi tính toán[cite: 67, 68].
3.  [cite_start]**Optimizer:** `torch.optim.Adam` được sử dụng[cite: 65].
4.  Mô hình được huấn luyện trong **5 epochs**.

---

## 2. Kết quả Đánh giá (Task 5)

### Đánh giá Độ chính xác trên Tập Validation

[cite_start]Độ chính xác được tính trên các token **không phải padding** của tập validation[cite: 81, 82].

| Chỉ số | Giá trị |
| :--- | :--- |
| **Độ chính xác trên tập validation (Accuracy)** | **93.14%** |

### Ví dụ Dự đoán Câu mới

[cite_start]Đã chạy hàm `predict_sentence` với câu thử nghiệm mới để kiểm tra khả năng khái quát hóa của mô hình[cite: 87]:

| Câu gốc: | "VNU University is located in Hanoi" |
| :--- | :--- |
| VNU | B-ORG |
| University | I-ORG |
| is | O |
| located | O |
| in | O |
| Hanoi | B-LOC |

**Nhận xét:** Mô hình đã nhận diện chính xác "VNU University" là một thực thể Tổ chức (ORG) và "Hanoi" là một thực thể Địa điểm (LOC).

---

## 4. Nộp bài

**KẾT QUẢ THỰC HIỆN**

* **Độ chính xác trên tập validation:** **93.14%**
* **Ví dụ dự đoán câu mới:** Xem kết quả in ra cho câu "VNU University is located in Hanoi"