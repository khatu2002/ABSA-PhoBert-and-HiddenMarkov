import nltk
from nltk.data import find

# Download the wordnet resource if not already downloaded
try:
    find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')
#train_phobert_absa.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn
import torch
from torch.optim import AdamW
from data_utils import get_data_loaders
from evaluate_model import evaluate
import os

# Định nghĩa Focal Loss với alpha cho từng lớp
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha else [1, 1, 1]  # Đặt alpha mặc định cho từng lớp nếu không có
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Sử dụng alpha cho từng lớp
        at = torch.tensor(self.alpha, device=inputs.device)[targets]
        focal_loss = at * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Load PhoBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Sử dụng AdamW optimizer
optimizer = AdamW(model.parameters(), lr=5e-6)  # Giảm learning rate

# Cấu hình FocalLoss với alpha điều chỉnh cho lớp neutral
criterion = FocalLoss(alpha=[0.5, 2.0, 1.0], gamma=2.5)  # Tăng trọng số cho lớp neutral

# Hàm huấn luyện với lưu lại state_dict khi có độ chính xác cao nhất
def train(model, train_loader, test_loader, optimizer, criterion, epochs=5):
    best_accuracy = 0  # Theo dõi độ chính xác tốt nhất
    best_model_state = None  # Lưu lại state_dict tốt nhất

    # Lấy tên mô hình mà không có thư mục con
    model_name = model.config._name_or_path.split("/")[-1]

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # Đánh giá mô hình sau mỗi epoch
        accuracy = evaluate(model, test_loader)

        # Lưu lại state_dict nếu độ chính xác cao hơn độ chính xác tốt nhất
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()  # Lưu lại state_dict tốt nhất
            print(f"New best accuracy: {best_accuracy:.4f}")

            # Đảm bảo thư mục lưu trữ tồn tại
            if not os.path.exists('state_dict'):
                os.makedirs('state_dict', exist_ok=True)

            # Lưu state_dict với tên chứa thông tin độ chính xác
            save_path = os.path.join('state_dict', f'{model_name}_val_acc_{round(best_accuracy, 4)}.pth')
            torch.save(best_model_state, save_path)
            print(f"Best model saved at: {save_path}")

# Khởi tạo DataLoader và huấn luyện
train_loader, test_loader = get_data_loaders()
train(model, train_loader, test_loader, optimizer, criterion, epochs=5)