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