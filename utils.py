import os
import re
import string
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpamDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab or self.build_vocab(texts)
        self.max_length = max_length

    def build_vocab(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split())
        return {word: idx for idx, word in enumerate(vocab)}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = [self.vocab.get(word, 0) for word in text.split()]
        if len(encoded) < self.max_length:
            encoded += [0] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
        return torch.tensor(encoded), torch.tensor(label)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        out = self.fc(hidden)
        return self.sigmoid(out)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data():
    print("Preprocessing data...")
    os.makedirs('processed_data', exist_ok=True)
    ham_texts, spam_texts = [], []
    for i in range(1, 7):
        ham_path = f'./enron/enron{i}/ham/'
        spam_path = f'./enron/enron{i}/spam/'
        for filename in os.listdir(ham_path):
            with open(os.path.join(ham_path, filename), 'r', errors='ignore') as file:
                ham_texts.append(preprocess_text(file.read()))
        for filename in os.listdir(spam_path):
            with open(os.path.join(spam_path, filename), 'r', errors='ignore') as file:
                spam_texts.append(preprocess_text(file.read()))
    
    with open('./processed_data/ham.txt', 'w', encoding='utf-8') as f:
        for text in ham_texts:
            f.write(f"{text}\n")
    
    with open('./processed_data/spam.txt', 'w', encoding='utf-8') as f:
        for text in spam_texts:
            f.write(f"{text}\n")

    texts = ham_texts + spam_texts
    labels = [0] * len(ham_texts) + [1] * len(spam_texts)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42)

    with open('./processed_data/train.txt', 'w', encoding='utf-8') as f:
        for text, label in zip(train_texts, train_labels):
            f.write(f"{label}\t{text}\n")

    with open('./processed_data/test.txt', 'w', encoding='utf-8') as f:
        for text, label in zip(test_texts, test_labels):
            f.write(f"{label}\t{text}\n")
