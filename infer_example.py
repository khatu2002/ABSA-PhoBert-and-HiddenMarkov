import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from joblib import load
import numpy as np

# Đường dẫn tới mô hình tích hợp, từ điển khía cạnh và vectorizer
MODEL_PATH = "state_dict/combined_model_combined_acc_0.8435.pth"
ASPECT_DICT_PATH = "aspect_dict.txt"
NAIVE_BAYES_MODEL_PATH = "train_model/naive_bayes_model.pkl"
VECTORIZER_PATH = "train_model/tfidf_vectorizer.pkl"

# Load từ điển khía cạnh
def load_aspect_dict(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        aspect_keywords = f.read().splitlines()
    return sorted(set(aspect.lower() for aspect in aspect_keywords), key=len, reverse=True)

# Load PhoBERT tokenizer và model
tokenizer = AutoTokenizer.from_pretrained("train_model")  # Tải tokenizer từ thư mục train_model
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))  # Tải trọng số từ file tích hợp
model.eval()

# Load Naive Bayes model
naive_bayes_model = load(NAIVE_BAYES_MODEL_PATH)

# Load TfidfVectorizer
vectorizer = load(VECTORIZER_PATH)

# Hàm nhận diện và nhóm khía cạnh với phần mô tả liên quan
def identify_aspects(sentence, aspect_dict):
    sentence = sentence.lower()
    for keyword in aspect_dict:
        if re.search(rf'\b{keyword}\b', sentence):
            return {keyword: sentence}  # Trả về khía cạnh đầu tiên tìm thấy
    return {}

# Hàm dự đoán cảm xúc với sự kết hợp PhoBERT và Naive Bayes
def predict_sentiment(sentence, aspect):
    # Bước 1: Chuẩn bị đầu vào cho PhoBERT
    inputs = tokenizer.encode_plus(
        f"Về khía cạnh {aspect}, {sentence}",
        add_special_tokens=True,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Bước 2: Dự đoán với PhoBERT
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    phoBERT_logits = outputs.logits.numpy().flatten()

    # Bước 3: Sử dụng TfidfVectorizer để trích xuất đặc trưng cho Naive Bayes
    tfidf_features = vectorizer.transform([sentence]).toarray()  # Chuyển câu thành vector TF-IDF

    # Chuyển các đặc trưng PhoBERT thành định dạng tương thích với Naive Bayes
    nb_features = np.hstack([phoBERT_logits.reshape(1, -1), tfidf_features])  # Ghép các đặc trưng PhoBERT và TF-IDF lại

    # Adjusting to match the number of features
    # If nb_features > 4526, truncate it, or if nb_features < 4526, pad it
    expected_features = 4526
    if nb_features.shape[1] > expected_features:
        nb_features = nb_features[:, :expected_features]  # Truncate if more features than expected
    elif nb_features.shape[1] < expected_features:
        padding = np.zeros((nb_features.shape[0], expected_features - nb_features.shape[1]))  # Pad if fewer features
        nb_features = np.hstack([nb_features, padding])  # Add padding

    # Bước 4: Dự đoán bằng Naive Bayes
    nb_prediction = naive_bayes_model.predict(nb_features)[0]

    # Bước 5: Kết hợp kết quả từ PhoBERT và Naive Bayes
    phoBERT_class = np.argmax(phoBERT_logits)
    if phoBERT_class == nb_prediction:
        final_class = phoBERT_class
    else:
        # Weighted voting (tùy chọn: điều chỉnh trọng số 0.6 và 0.4 tùy ý)
        final_class = phoBERT_class if phoBERT_logits[phoBERT_class] * 0.6 > 0.4 else nb_prediction

    # Bước 6: Xác định nhãn cảm xúc từ lớp dự đoán cuối cùng
    sentiment = "Positive" if final_class == 2 else "Neutral" if final_class == 1 else "Negative"
    return sentiment


# Tách câu dựa trên các dấu phẩy và các từ nối
def split_sentence(sentence):
    splitters = [" nhưng ", " tuy nhiên ", " mặc dù ", " song ", ",", ";", "."]
    parts = [sentence]
    for splitter in splitters:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(splitter))
        parts = new_parts
    return [part.strip() for part in parts if part.strip()]

# Hàm chính để xử lý câu đầu vào
def main(sentence):
    aspect_dict = load_aspect_dict(ASPECT_DICT_PATH)
    split_words = [" nhưng ", " tuy nhiên ", " mặc dù ", " song ", ";", ","]

    if any(word in sentence.lower() for word in split_words):
        sentence_parts = split_sentence(sentence.lower())
        for part in sentence_parts:
            aspects = identify_aspects(part, aspect_dict)
            if aspects:
                main_aspect = list(aspects.keys())[0]
                main_phrase = aspects[main_aspect]
                sentiment = predict_sentiment(main_phrase, main_aspect)
                print(f"Extracted aspect: {main_aspect}")
                print(f"Combined phrases: {main_phrase}")
                print(f"Prediction for aspect ('{main_aspect}'): {sentiment}")
    else:
        aspects = identify_aspects(sentence.lower(), aspect_dict)
        if aspects:
            main_aspect = list(aspects.keys())[0]
            main_phrase = aspects[main_aspect]
            sentiment = predict_sentiment(main_phrase, main_aspect)
            print(f"Extracted aspect: {main_aspect}")
            print(f"Combined phrases: {main_phrase}")
            print(f"Prediction for aspect ('{main_aspect}'): {sentiment}")

if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:])
    main(sentence)
