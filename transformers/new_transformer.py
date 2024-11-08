import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers.tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.load()
words = tokenizer.get_words()

# Load processed data
train_masked = np.load('train_masked.npy')
train_labels = np.load('train_labels.npy')
val_masked = np.load('val_masked.npy')
val_labels = np.load('val_labels.npy')
test_masked = np.load('test_masked.npy')
test_labels = np.load('test_labels.npy')
vocab_size = int(np.load('vocab_size.npy'))
train_segments = np.load('train_segments.npy')
val_segments = np.load('val_segments.npy')
test_segments = np.load('test_segments.npy')

# After loading data
print("Train segments shape:", train_segments.shape)
input_ids = torch.tensor(train_segments, dtype=torch.long)
print("Input ids shape:", input_ids.shape)
print("Max value in input_ids:", input_ids.max().item())
print("Min value in input_ids:", input_ids.min().item())

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=8, max_seq_length=150):
        super().__init__()
        print(f"Initializing transformer with vocab_size={vocab_size}, max_seq_length={max_seq_length}")
        self.word_embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.prediction_head = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_length = max_seq_length
        
    def forward(self, x):
        print("Input shape:", x.shape)
        # Get word embeddings
        word_embeddings = self.word_embedding(x)
        
        # Create position indices
        positions = torch.arange(x.size(-1), device=x.device)
        print("Positions shape:", positions.shape)
        print("Max position:", positions.max().item())
        
        # Add position embeddings
        pos_embeddings = self.position_embedding(positions)
        
        # Combine word and position embeddings
        embeddings = word_embeddings + pos_embeddings.unsqueeze(0)
        embeddings = self.layer_norm(embeddings)
        
        # Attention mechanism
        attention_scores = torch.matmul(embeddings, embeddings.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_weights, embeddings)
        
        # Predict
        logits = self.prediction_head(hidden_states)
        
        return logits

# Create input tensor
input_ids = torch.tensor(train_segments, dtype=torch.long)
print("Original input_ids:", input_ids.tolist())

# Create input tensor for masked sentence training
masked_input_ids = torch.tensor(train_masked, dtype=torch.long)
masked_labels = torch.tensor(train_labels, dtype=torch.long)

# Initialize model
model = SimpleTransformer(vocab_size, embedding_dim=8, max_seq_length=150)


def train_model(model, input_ids, labels=None, epochs=200, masked=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    

    for epoch in range(epochs):  
        model.train()
        
        if masked:
            # Use masked inputs and labels
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        else:
            # Use full sentence inputs
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), input_ids.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"\nEpoch {epoch}, Loss: {loss.item():.4f}")

            if masked:
                probs = torch.softmax(logits, dim=-1)
                pred_indices = torch.argmax(logits, dim=-1).cpu().numpy()  # Convert to numpy array
                pred_words = [words[idx] for idx in pred_indices.flatten()]  # Flatten and convert to list

                for i, (mask, label) in enumerate(zip(masked_input_ids, labels)):
                    if label != -100:  # if position was masked
                        print(f"\nPosition {i}:")
                        print(f"True word: {words[label]}")
                        print(f"Predicted: {pred_words[i]}")
                        print("Probabilities:")
                        # Show probability for each possible word
                        word_probs = probs[i].tolist()
                        for word, prob in zip(words, word_probs):
                            print(f"{word}: {prob:.3f}")
    

# Phase 1: Train on full sentences
train_model(model, torch.tensor(train_segments, dtype=torch.long), epochs=100, masked=False)

# Phase 2: Train on masked sentences
train_model(model, torch.tensor(train_masked, dtype=torch.long), torch.tensor(train_labels, dtype=torch.long), epochs=500, masked=True)
