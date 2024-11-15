import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from data_utils import tokenizer

def evaluate_combined(phobert_model, hmm_model, test_loader, hmm_vocab, device, phobert_weight=0.7, hmm_weight=0.3):
    phobert_model.to(device)
    phobert_model.eval()

    total_correct = 0
    total_samples = 0
    all_labels = []
    all_combined_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # PhoBERT prediction
            outputs = phobert_model(input_ids=input_ids, attention_mask=attention_mask)
            phobert_preds = torch.argmax(outputs.logits, dim=-1)
            probs_phobert = torch.softmax(outputs.logits, dim=-1).cpu().numpy()  # Convert to NumPy

            # HMM prediction
            hmm_preds = []
            hmm_probs = []
            for i in range(input_ids.size(0)):
                sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True).split()
                seq = [hmm_vocab.vocabulary_.get(word, -1) for word in sentence if word in hmm_vocab.vocabulary_]

                if -1 in seq:  # Skip sequences with unknown words in HMM vocab
                    hmm_preds.append(1)
                    hmm_probs.append([1.0, 0.0, 0.0])  # Default probability for Naive Bayes
                else:
                    seq_array = np.array(seq).reshape(1, -1)
                    if seq_array.shape[1] < 4590:
                        padding = np.zeros((1, 4590 - seq_array.shape[1]))
                        seq_array = np.hstack((seq_array, padding))
                    _, hidden_states = hmm_model.decode(seq_array)
                    hmm_pred = np.bincount(hidden_states).argmax()
                    hmm_preds.append(hmm_pred)
                    hmm_probs.append([0.3, 0.4, 0.3])  # Example probabilities

            # Combine predictions using weighted voting with highest confidence
            combined_preds = []
            for i in range(len(phobert_preds)):
                phobert_max_prob = max(probs_phobert[i])
                hmm_max_prob = max(hmm_probs[i])

                if phobert_max_prob > hmm_max_prob:
                    combined_preds.append(phobert_preds[i].item())  # Convert to Python int
                else:
                    combined_preds.append(hmm_preds[i])

            # Append predictions and labels for the combined model
            all_labels.extend(labels.cpu().numpy().tolist())
            all_combined_preds.extend(combined_preds)

            # Calculate accuracy
            correct = (torch.tensor(combined_preds).cpu() == labels.cpu()).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Combined Model Accuracy: {accuracy * 100:.2f}%")

    # Convert predictions and labels to NumPy arrays for classification report
    all_labels = np.array(all_labels)
    all_combined_preds = np.array(all_combined_preds)

    # Print classification report
    print("\nCombined Model Classification Report:")
    print(classification_report(all_labels, all_combined_preds, target_names=["negative", "neutral", "positive"]))

    return accuracy
