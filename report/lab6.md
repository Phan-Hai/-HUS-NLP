# BÀI 1: KHÔI PHỤC MASKED TOKEN

1. Mô hình đã dự đoán đúng từ "capital" không?

   Có, mô hình đã dự đoán đúng từ "capital" với độ tin cậy rất cao 
   (99%). Điều này cho thấy mô hình BERT đã học được mối quan hệ 
   ngữ nghĩa giữa "Hanoi" và "capital of Vietnam" rất tốt.

2. Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?
   - BERT sử dụng cơ chế Self-Attention hai chiều (bidirectional), cho phép mô hình 
     xem xét cả các từ đứng trước và sau token [MASK].
   - Trong câu "Hanoi is the [MASK] of Vietnam", BERT có thể nhìn thấy cả "Hanoi" 
     (trước) và "of Vietnam" (sau) để đưa ra dự đoán chính xác.
   - BERT được huấn luyện trước (pre-trained) với tác vụ Masked Language Modeling (MLM), 
     nên nó rất giỏi trong việc dự đoán từ bị che giấu dựa trên ngữ cảnh xung quanh.
   - Các mô hình Decoder-only (như GPT) chỉ nhìn một chiều (từ trái sang phải) nên 
     không thể tận dụng thông tin từ các từ phía sau token [MASK].

# BÀI 2: DỰ ĐOÁN TỪ TIẾP THEO

1. Kết quả sinh ra có hợp lý không?

   Kết quả thường khá hợp lý về mặt ngữ pháp và có tính mạch lạc. 
   Tuy nhiên, nội dung có thể không hoàn toàn chính xác về mặt thực tế hoặc 
   có thể hơi chung chung. Mô hình GPT-2 mặc định được huấn luyện trên dữ liệu 
   tiếng Anh nên nó có khả năng sinh văn bản tiếng Anh tự nhiên và có ngữ nghĩa.

2. Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ này?
    
   - GPT được thiết kế với cơ chế Self-Attention một chiều (unidirectional/causal), 
     chỉ xem xét các token đã xuất hiện trước đó. Điều này phù hợp với tác vụ 
     sinh văn bản tuần tự (từ trái sang phải).
   - GPT được huấn luyện với mục tiêu Next Token Prediction, tức là dự đoán token 
     tiếp theo dựa trên các token trước đó. Đây chính là bản chất của tác vụ text generation.
   - Kiến trúc Decoder-only cho phép mô hình sinh từng token một cách tự hồi quy 
     (autoregressive), phù hợp với việc tạo ra chuỗi văn bản dài và mạch lạc.
   - Các mô hình Encoder-only (như BERT) không phù hợp vì chúng được thiết kế để 
     hiểu ngữ cảnh (understanding), không phải sinh văn bản (generation). BERT nhìn 
     cả hai chiều nên không thể sinh văn bản tuần tự một cách tự nhiên.

# BÀI 3: TÍNH TOÁN VECTOR BIỂU DIỄN CỦA CÂU

1. Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với
tham số nào của mô hình BERT?

Con số này tương ứng với tham số hidden_size của mô hình BERT-base. Đây là số chiều của vector embedding mà mỗi token được biểu diễn trong các hidden layers. Từ kết quả last_hidden_state.shape = [1, 8, 768], ta thấy mỗi token trong 8 tokens của câu đều được biểu diễn bằng vector 768 chiều.

2. Tại sao chúng ta cần sử dụng attention_mask khi thực hiện Mean Pooling?

Chúng ta cần attention_mask để loại bỏ padding tokens khỏi phép tính trung bình.
Khi xử lý batch nhiều câu có độ dài khác nhau, tokenizer sẽ đệm (padding) các câu ngắn hơn. Padding tokens không mang ý nghĩa ngữ nghĩa, nếu tính vào trung bình sẽ làm méo mó vector biểu diễn.