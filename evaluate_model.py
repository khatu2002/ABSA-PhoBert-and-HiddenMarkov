#evaluate_model.py
from sklearn.metrics import accuracy_score, classification_report
import torch
import numpy as np

# Evaluate PhoBERT
def evaluate(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = attention_mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["negative", "neutral", "positive"])
    
    print(f"PhoBERT Accuracy: {accuracy * 100:.2f}%")
    print(report)
    
    return accuracy

# Evaluate HMM
def evaluate_hmm(model, vectorizer, sentences, labels):
    y_true, y_pred = [], []

    # Transform the sentences using the vectorizer
    X_test = vectorizer.transform(sentences).toarray()

    for x, label in zip(X_test, labels):
        # Reshape x to 2D (1, n_features) before passing to decode
        _, hidden_states = model.decode(x.reshape(1, -1))  # Decode the sequence

        # Get the predicted label (based on the most common hidden state)
        predicted_label = np.bincount(hidden_states).argmax()

        y_true.append(label)
        y_pred.append(predicted_label)

    # Compute accuracy and classification report
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"])

    print(f"HMM Accuracy: {accuracy * 100:.2f}%")
    print(report)
    return accuracy

