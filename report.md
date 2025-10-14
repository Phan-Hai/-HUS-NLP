# NLP Lab Report: Lab 1 & Lab 2

## Lab 1: Text Tokenization

### Mục tiêu
Triển khai bước tiền xử lý cơ bản trong NLP: **tokenization**:
- Cài đặt `SimpleTokenizer`.
- Cài đặt `RegexTokenizer` dựa trên biểu thức chính quy.
- Thử nghiệm tokenizer trên các câu mẫu và trên dataset **UD_English-EWT**.

### Công việc thực hiện
1. **SimpleTokenizer**
   - Chuyển tất cả chữ về lowercase.
   - Tách từ dựa trên khoảng trắng.
   - Xử lý dấu câu cơ bản (`. , ? !`) bằng cách tách riêng khỏi từ.
   
2. **RegexTokenizer**
   - Sử dụng regex `\w+|[^\w\s]` để trích xuất token.
   - Phương pháp này mạnh mẽ hơn trong việc xử lý các trường hợp phức tạp (ví dụ: từ viết tắt, apostrophe).

3. **Tokenization trên UD_English-EWT**
   - Lấy 500 ký tự đầu tiên của dataset.
   - So sánh output giữa `SimpleTokenizer` và `RegexTokenizer`.

### Kết quả chạy code

#### Tokenizing các câu mẫu:
```csharp
Original: Hello, world! This is a test.
SimpleTokenizer: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer:   ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Original: NLP is fascinating... isn't it?
SimpleTokenizer: ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
RegexTokenizer:   ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Original: Let's see how it handles 123 numbers and punctuation!
SimpleTokenizer: ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
RegexTokenizer:   ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
```

## Lab 2: Count Vectorization

### Mục tiêu
**Biểu diễn văn bản thành vector số** bằng **Bag-of-Words**:
- Sử dụng tokenizer từ Lab 1.
- Cài đặt `CountVectorizer` để tạo vocabulary và document-term matrix.

### Công việc thực hiện
1. **Vectorizer Interface**
   - Định nghĩa interface với các phương thức: `fit`, `transform`, `fit_transform`.

2. **CountVectorizer**
   - Nhận một instance của tokenizer.
   - Thu thập vocabulary từ corpus.
   - Chuyển mỗi document thành vector đếm tần suất token.

3. **Evaluation**
   - Dùng `RegexTokenizer` để token hóa corpus mẫu.
   - Chạy `fit_transform` trên corpus:

```csharp
Vocabulary: {'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}

Document-Term Matrix:
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```

# NLP Lab Report: Lab 3

## I. Giải thích các bước thực hiện

Trong bài thực hành này, em thực hiện các bước sau để nghiên cứu và trực quan hóa các từ vector từ các mô hình embedding:

### Bước 1: Sử dụng mô hình pre-trained (task 1 2)
1. **Tải mô hình pre-trained từ Gensim API**:
   - Sử dụng lớp `WordEmbedder` để tải các mô hình sẵn có như `glove-wiki-gigaword-50`. Lớp được xây dựng ở `src\representations\word_embedder.py`
   - Mục đích: so sánh các từ đồng nghĩa, tính độ tương đồng giữa các từ, và dùng vector từ mô hình đã huấn luyện trước đó để trực quan hóa.
2. **Trích xuất vector và tính tương đồng**:
   Các yêu cầu về đánh giá được viết ở `lab3\test\test_b3.py`
   - Lấy vector cho từng từ bằng hàm `get_vector`.
   - Tính độ tương đồng giữa các từ với `get_similarity`.
   - Tìm các từ gần nghĩa nhất với `get_most_similar`.
   - Hàm embed một document với `embed_document`.

### Bước 2: Huấn luyện mô hình Word2Vec trên tập dữ liệu nhỏ (Task 3)
Thực hiện ở `lab3\test\lab4_embedding_training_demo.py`
1. **Chuẩn bị dữ liệu**:
   - Đọc tập dữ liệu văn bản nhỏ từ `en_ewt-ud-train.txt`.
   - Sử dụng lớp `SentenceStream` để trả về từng câu đã được tiền xử lý (`simple_preprocess`) cho Word2Vec.
2. **Huấn luyện mô hình Word2Vec**:
   - Cấu hình: vector size = 100, window = 5, min_count = 3, workers = 8, sử dụng Skip-gram (`sg=1`).
   - Lưu mô hình sau khi huấn luyện để sử dụng lại.
3. **Demo một số phép toán embedding**:
   - Tìm các từ đồng nghĩa với `computer`.
   - Thực hiện phép toán tương tự analogies: `king - man + woman ≈ ?`.

### Bước 3: Huấn luyện mô hình Word2Vec trên tập dữ liệu lớn với Spark (Task 4)
Thực hiên ở `lab3\test\lab4_spark_word2vec_demo.py`
1. **Khởi tạo SparkSession và đọc dữ liệu lớn**:
   - Dữ liệu JSON lớn (`c4-train`) được đọc vào Spark DataFrame.
   - Lọc bỏ các dòng null, chuẩn hóa chữ thường và loại bỏ ký tự đặc biệt.
2. **Tiền xử lý và tách từ**:
   - Sử dụng `Tokenizer` của Spark ML để tách câu thành các từ.
3. **Huấn luyện Word2Vec**:
   - vectorSize = 100, minCount = 5.
   - Sau khi huấn luyện, lưu mô hình và tìm các từ đồng nghĩa với từ `computer`.
   
### Bước 4: Giảm chiều và trực quan hóa từ vector (Task 5)
Thực hiện ở `lab3\b3.ipynb`
1. **Load mô hình GloVe 50 chiều**:
   - Đọc file `.txt` GloVe, lưu các từ và vector vào dictionary.
   - Lấy mẫu 40.000 từ ngẫu nhiên để giảm tải trực quan hóa.
2. **Giảm chiều**:
   - Sử dụng **PCA** để giảm từ 50 chiều xuống 2 chiều, giữ cấu trúc tổng thể.
   - Sử dụng **t-SNE** để giảm từ 50 chiều xuống 2 chiều, tập trung vào mối quan hệ cục bộ giữa các từ.
3. **Trực quan hóa**:
   - Vẽ biểu đồ scatter plot cho cả PCA và t-SNE.
   - Ghi nhãn các từ đầu tiên (100 từ cho PCA, 200 từ cho t-SNE) để quan sát trực quan.
   - So sánh trực quan giữa hai phương pháp giảm chiều.

## II. Hướng dẫn chạy code

Để thực thi lại các bài tập và quan sát kết quả, hãy làm theo hướng dẫn sau:

### Task 1 & 2: Sử dụng mô hình pre-trained
- **File thực thi:** `lab3/test/test_b3.py`
- **Hướng dẫn:**
  1. Kích hoạt môi trường ảo (nếu có):
     ```bash
     & "D:/10. ky1nam4/NLP/venv/Scripts/Activate.ps1"
     ```
  2. Chạy script:
     ```bash
     python lab3/test/test_b3.py
     ```
  3. Quan sát các kết quả:
     - Các vector từ được tải từ mô hình pre-trained.
     - Danh sách các từ đồng nghĩa với từ nhập.
     - Độ tương đồng giữa các từ.
  4. Kết quả:
      ```bash
      Đang tải mô hình 'glove-wiki-gigaword-50'...
      Mô hình 'glove-wiki-gigaword-50' đã được tải.

      Vector for 'king':
      [ 0.50451   0.68607  -0.59517  -0.022801  0.60046  -0.13498  -0.08813
      0.47377  -0.61798  -0.31012  -0.076666  1.493    -0.034189 -0.98173
      0.68229   0.81722  -0.51874  -0.31503  -0.55809   0.66421   0.1961
      -0.13495  -0.11476  -0.30344   0.41177  -2.223    -1.0756   -1.0783
      -0.34354   0.33505   1.9927   -0.04234  -0.64319   0.71125   0.49159
      0.16754   0.34344  -0.25663  -0.8523    0.1661    0.40102   1.1685
      -1.0137   -0.21585  -0.15155   0.78321  -0.91241  -1.6106   -0.64426
      -0.51042 ]

      Similarity(king, queen): 0.78390425
      Similarity(king, man): 0.53093773

      Most similar to 'computer':
      computers       0.9165
      software        0.8815
      technology      0.8526
      electronic      0.8126
      internet        0.8060
      computing       0.8026
      devices         0.8016
      digital         0.7992
      applications    0.7913
      pc              0.7883

      Document embedding for: "The queen rules the country."
      [ 0.04564168  0.36530998 -0.55974334  0.04014383  0.09655549  0.15623933
      -0.33622834 -0.12495166 -0.01031508 -0.5006717   0.18690467  0.17482166
      -0.268985   -0.03096624  0.36686516  0.29983264  0.01397333 -0.06872118
      -0.3260683  -0.210115    0.16835399 -0.03151734 -0.06204716  0.04301083
      -0.06958768 -1.7792168  -0.54365396 -0.06104483 -0.17618     0.009181
      3.3916333   0.08742473 -0.4675417  -0.213435    0.02391887 -0.04470453
      0.20636833 -0.12902866 -0.28527132 -0.2431805  -0.3114423  -0.03833717
      0.11977985 -0.01418401 -0.37086335  0.22069354 -0.28848937 -0.36188802
      -0.00549529 -0.46997246]

      Vector dimension: 50
      ```

### Task 3: Huấn luyện Word2Vec trên tập dữ liệu nhỏ
- **File thực thi:** `lab3/test/lab4_embedding_training_demo.py`
- **Hướng dẫn:**
  1. Chạy script:
     ```bash
     python lab3/test/lab4_embedding_training_demo.py
     ```
  2. Kết quả bao gồm:
     - Mô hình Word2Vec được huấn luyện và lưu trong thư mục `results`.
     - Demo tìm từ đồng nghĩa (`most_similar`) và phép toán analogies (`king - man + woman ≈ ?`).
  3. Kết quả
   ```bash
   Training Word2Vec model...
   Huấn luyện xong!
   Done saving
   Tìm các từ gần nghĩa với 'computer'
   begin           0.9957
   seek            0.9952
   ability         0.9951
   admissions      0.9948
   vehicle         0.9947
   Ví dụ: king - man + woman ≈ ?
   Kết quả gần đúng: launcher (0.9915)
   ```

### Task 4: Huấn luyện Word2Vec trên tập dữ liệu lớn với Spark
- **File thực thi:** `lab3/test/lab4_spark_word2vec_demo.py`
- **Hướng dẫn:**
  1. Đảm bảo đã cài đặt PySpark và Spark chạy bình thường.
  2. Chạy script:
     ```bash
     python lab3/test/lab4_spark_word2vec_demo.py
     ```
  3. Kết quả bao gồm:
     - Mô hình Word2Vec Spark được lưu vào `results/word2vec_spark_model.model`.
     - In ra 5 từ đồng nghĩa gần nhất với từ `computer`.
  ```bash
  5 từ đồng nghĩa với 'computer':
   +-------+------------------+
   |   word|        similarity|
   +-------+------------------+
   |desktop|0.6910857558250427|
   |   198x|0.6837746500968933|
   | laptop|0.6687031388282776|
   |     pc|0.6550753116607666|
   | tablet|0.6527244448661804|
   +-------+------------------+

  ```

### Task 5: Giảm chiều và trực quan hóa
- **File thực thi:** `lab3/b3.ipynb`
- **Hướng dẫn:**
  1. Mở file `.ipynb` trong Jupyter Notebook hoặc VSCode.
  2. Chạy từng cell theo thứ tự:
     - Load mô hình GloVe 50D.
     - Giảm chiều bằng PCA và t-SNE.
     - Vẽ trực quan hóa scatter plot cho PCA và t-SNE.
  3. Quan sát biểu đồ để phân tích cụm từ, mối quan hệ giữa các từ.

## III. Phân tích kết quả

### 3.1 Độ tương đồng và các từ đồng nghĩa từ mô hình pre-trained

- Mô hình pre-trained phản ánh khá tốt mối quan hệ giữa các từ liên quan về giới tính và vai trò (king ↔ queen, king ↔ man).  
- Giá trị similarity cao cho thấy các từ cùng ngữ nghĩa gần nhau trong không gian vector.

- Các từ đồng nghĩa gần nhất với "computer" gồm:
  computers, software, technology, electronic, internet, computing, devices, digital, applications, pc.

### 3.2 So sánh với mô hình tự huấn luyện

- Mô hình Word2Vec trên tập dữ liệu nhỏ (Gensim):
  - Các từ đồng nghĩa tìm được gần như không liên quan đến "computer": wheel, girlfriend, image, favorite, parking.  
  - Phép toán analogies `king - man + woman ≈ thin` cho kết quả không chính xác.  
  - Nguyên nhân: tập dữ liệu nhỏ, không đủ đa dạng, nên embedding kém chất lượng.

- Mô hình Word2Vec trên tập dữ liệu lớn (Spark):
  - Các từ đồng nghĩa tìm được hợp lý hơn: desktop, laptop, 198x, computers, pc.  
  - Chất lượng embedding tốt hơn so với mô hình tập nhỏ, mặc dù similarity thấp hơn pre-trained.

### 3.3 Phân tích biểu đồ trực quan hóa

- **Cấu trúc tổng quan:**
  - Biểu đồ PCA: các từ tập trung khá dày đặc ở trung tâm, phân bố đồng đều nhưng cụm từ rõ ràng ít thấy. PCA giữ được cấu trúc tổng thể, nhưng khó thấy các mối quan hệ cục bộ.
  - Biểu đồ t-SNE: các từ phân bố rộng hơn, các cụm từ cùng ngữ nghĩa xuất hiện gần nhau rõ ràng hơn. t-SNE tập trung vào mối quan hệ cục bộ, làm nổi bật các nhóm từ.

- **Các cụm từ thú vị:**
  - Các từ liên quan đến công nghệ như "computer", "software", "internet", "pc" xuất hiện gần nhau trên t-SNE.  
  - Các từ liên quan tên người (ví dụ: "khristine", "ryley", "arianna") cũng tập trung thành cụm riêng.  
  - Các từ chỉ hành động hoặc trạng thái (ví dụ: "undertake", "mobilised", "starved") có xu hướng nhóm lại với nhau.

- **Nhận xét:**
  - t-SNE hiển thị cụm từ rõ ràng hơn PCA, giúp quan sát các mối quan hệ cục bộ giữa từ và nhóm từ đồng nghĩa.  
  - PCA giữ cấu trúc toàn cục nhưng làm mất chi tiết cục bộ.  
  - Cụm từ xuất hiện gần nhau như kỳ vọng: các từ cùng chủ đề hoặc cùng ngữ nghĩa thường nằm cạnh nhau trên t-SNE, cho thấy embedding GloVe đã học được mối quan hệ ngữ nghĩa giữa các từ.

> Kết luận: trực quan hóa t-SNE phù hợp để quan sát mối quan hệ cục bộ và cụm từ đồng nghĩa, trong khi PCA thích hợp để nhìn tổng thể cấu trúc vector.

## IV. Khó khắn khi thực hiện
- Phần cài Spark hơi mất thời gian vì phải cài lại Java. 
- Có một số lỗi khi em lưu model ở ví dụ với Spark. Spark đang cố lưu mô hình ra ổ đĩa (save), và Spark sử dụng Hadoop FileSystem để thao tác file, nhưng trên Windows nó cần file winutils.exe để quản lý quyền file. Hiện tại Spark không tìm thấy file này. Do em chưa cài Hadoop

## V. Trích dẫn tài liệu:
Ngoài các thư viện em sử dụng khi code thì còn có sự hỗ trợ của ChatGPT và Copilot