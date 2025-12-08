# Báo cáo Thực hành: Dependency Parsing với spaCy

## 1. Giới thiệu
Phân tích cú pháp phụ thuộc (Dependency Parsing) là kỹ thuật xác định cấu trúc ngữ pháp của câu bằng cách mô hình hóa quan hệ giữa các từ theo cặp head – dependent.

---

## 2. Chuẩn bị môi trường

    pip install -U spacy
    python -m spacy download en_core_web_md

    import spacy
    from spacy import displacy

    nlp = spacy.load("en_core_web_md")

---

## 3. Phân tích và trực quan hóa dependency

    text = "The quick brown fox jumps over the lazy dog."
    doc = nlp(text)
    displacy.serve(doc, style="dep")

Kết quả:
- ROOT: jumps
- nsubj: fox
- prep: over

---

## 4. Trích xuất dependency của từng token

    text = "Apple is looking at buying U.K. startup for $1 billion"
    doc = nlp(text)
    for token in doc:
        children = [child.text for child in token.children]
        print(token.text, token.dep_, token.head.text, children)

Kết luận:
- ROOT: looking
- pcomp: buying
- dobj: startup
- pobj: billion

---

## 5. Trích xuất Subject – Verb – Object

    text = "The cat chased the mouse and the dog watched them."
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "VERB":
            subject = ""
            obj = ""
            for child in token.children:
                if child.dep_ == "nsubj": subject = child.text
                if child.dep_ == "dobj": obj = child.text
            if subject and obj:
                print(subject, token.text, obj)

Kết quả:
    cat chased mouse

---

## 6. Trích xuất tính từ bổ nghĩa danh từ

    text = "The big, fluffy white cat is sleeping on the warm mat."
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "NOUN":
            adjectives = []
            for child in token.children:
                if child.dep_ == "amod": adjectives.append(child.text)
            if adjectives:
                print(token.text, adjectives)

Kết quả:
    cat ['big', 'white']
    mat ['warm']

---

## 7. Tìm động từ chính (ROOT)

    def find_main_verb(doc):
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token

    text = "The quick brown fox jumps over the lazy dog."
    doc = nlp(text)
    main_verb = find_main_verb(doc)
    print(main_verb.text)

Kết quả:
    jumps

---

## 8. Đường đi tới ROOT

    def get_path_to_root(token):
        path = [token]
        while token.head != token:
            token = token.head
            path.append(token)
        return path

    text = "Apple is looking at buying U.K. startup for $1 billion"
    doc = nlp(text)
    startup_token = doc[6]
    path = get_path_to_root(startup_token)
    print([t.text for t in path])

Kết quả:
    ['startup', 'buying', 'at', 'looking']

---

## 9. Kết luận

Qua buổi thực hành, tôi đã:
- hiểu dependency tree và ý nghĩa head–dependent
- xác định ROOT
- trích xuất quan hệ S-V-O
- tìm bổ nghĩa của danh từ thông qua amod
- lần theo quan hệ đến ROOT để phân tích cấu trúc câu

Ứng dụng thực tế:
- trích xuất quan hệ (Relation Extraction)
- hệ thống hỏi đáp
- phân tích ngữ nghĩa
- khai thác dữ liệu pháp luật, trợ lý RAG

---
