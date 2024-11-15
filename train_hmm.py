# train_hmm.py
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from hmmlearn import hmm
import pandas as pd
from evaluate_model import evaluate_hmm 

def load_hmm_data(filepath):
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath, sep='\t', header=None, names=['sentence', 'aspect', 'sentiment'])
    
    # Ensure that sentences are in string format (not lists of words)
    sentences = data['sentence'].tolist()
    labels = data['sentiment'].tolist()
    
    
    return sentences, labels

def train_hmm(sentences, labels, n_components=3):
    
    # Create a bag-of-words model for the sentences
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences).toarray()  # Transform sentences to event counts (BOW)

    # Make sure the number of samples matches the labels
    assert len(X) == len(labels), f"Number of samples in X ({len(X)}) does not match labels ({len(labels)})"
    
    # Fit the HMM model
    model = hmm.MultinomialHMM(n_components=n_components)
    model.fit(X)  # Train the model on the event counts
    
    print("HMM training complete.")
    return model, vectorizer

# Main execution
if __name__ == "__main__":
    # Load training data
    sentences, labels = load_hmm_data('train_final_cleaned.tsv')
    
    # Train HMM model
    model, vectorizer = train_hmm(sentences, labels)
    
    # Load test data (you can use the same function to load the test set)
    test_sentences, test_labels = load_hmm_data('test_final_cleaned.tsv')
    
    # Evaluate the model on the test set
    evaluate_hmm(model, vectorizer, test_sentences, test_labels)