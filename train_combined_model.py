# train_combined_model.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from hmmlearn import hmm
import numpy as np
from data_utils import get_data_loaders
from train_hmm import train_hmm, load_hmm_data
from evaluate_combined import evaluate_combined
from evaluate_model import evaluate, evaluate_hmm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained PhoBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
phobert_model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base")
phobert_model.to(device)  # Move model to device

# Load and prepare data for HMM training
hmm_train_sentences, hmm_train_labels = load_hmm_data('train_final_cleaned.tsv')
hmm_test_sentences, hmm_test_labels = load_hmm_data('test_final_cleaned.tsv')

# Train HMM model
hmm_model, hmm_vocab = train_hmm(hmm_train_sentences, hmm_train_labels, n_components=3)

# Get DataLoaders for PhoBERT
train_loader, test_loader = get_data_loaders(batch_size=16)

# Training function for combined model
def train_combined(phobert_model, hmm_model, hmm_vocab, test_loader):
    phobert_model.eval()
    best_accuracy = 0
    for epoch in range(5):
        print(f"Epoch {epoch + 1} - Evaluating Combined Model")
        
        # Evaluate PhoBERT
        print("\nEvaluating PhoBERT:")
        phobert_accuracy = evaluate(phobert_model, test_loader)
        
        # Evaluate HMM
        print("\nEvaluating HMM:")
        hmm_accuracy = evaluate_hmm(hmm_model, hmm_vocab, hmm_test_sentences, hmm_test_labels)
        
        # Evaluate Combined Model
        print("\nEvaluating Combined Model:")
        combined_accuracy = evaluate_combined(phobert_model, hmm_model, test_loader, hmm_vocab, device)
        
        # Track the best accuracy for the combined model
        if combined_accuracy > best_accuracy:
            best_accuracy = combined_accuracy
            print(f"New best combined accuracy: {best_accuracy * 100:.2f}%")
        
    print(f"\nBest Combined Model Accuracy: {best_accuracy * 100:.2f}%")

# Train and evaluate combined model
train_combined(phobert_model, hmm_model, hmm_vocab, test_loader)
