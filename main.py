import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from utils import preprocess_data, SpamDataset, LSTMModel, device
import pickle

def train_model():
    train_data = pd.read_csv('./processed_data/train.txt', sep='\t', header=None, names=['label', 'text'])
    vocab = SpamDataset(train_data['text'].tolist(), train_data['label'].tolist()).vocab
    train_dataset = SpamDataset(train_data['text'].tolist(), train_data['label'].tolist(), vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = LSTMModel(len(vocab)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device).long(), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}')

    torch.save(model.state_dict(), 'lstm_model.pth')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Convergence')
    plt.legend()
    plt.show()

def evaluate_model():
    test_data = pd.read_csv('./processed_data/test.txt', sep='\t', header=None, names=['label', 'text'])
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    test_dataset = SpamDataset(test_data['text'].tolist(), test_data['label'].tolist(), vocab)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LSTMModel(len(vocab)).to(device)
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device).long(), labels.to(device).float()
            outputs = model(texts)
            preds = (outputs.squeeze() > 0.5).int()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f'Test Accuracy: %{acc * 100}')
    print('Confusion Matrix:')
    print(cm)

    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Ham', 'Spam'], rotation=45)
    plt.yticks(tick_marks, ['Ham', 'Spam'])

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def main(train_model_flag):
    preprocess_data()
    if train_model_flag:
        train_model()
    evaluate_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_model', action='store_true', help='Train the model')
    args = parser.parse_args()
    main(args.train_model)
