import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
    

# 1. First let's create our sentence and vocabulary
sentence = "hello my name is lydia"
words = sentence.split()
vocab = {word: idx for idx, word in enumerate(words)}
vocab_size = len(vocab)

embedding_dim = 8

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length=5):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size + 1, embedding_dim)  # +1 for mask token
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.prediction_head = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        # Get word embeddings
        word_embeddings = self.word_embedding(x)
        
        # Add position embeddings
        positions = torch.arange(len(x)).to(x.device)
        pos_embeddings = self.position_embedding(positions)
        
        # Combine word and position embeddings
        embeddings = word_embeddings + pos_embeddings
        embeddings = self.layer_norm(embeddings)
        
        # Attention mechanism
        attention_scores = torch.matmul(embeddings, embeddings.transpose(0, 1))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_weights, embeddings)
        
        # Predict
        logits = self.prediction_head(hidden_states)
        
        return logits

# Create input tensor
input_ids = torch.tensor([vocab[word] for word in words])
print("Original input_ids:", input_ids.tolist())

# Initialize model
# model = SimpleTransformer(vocab_size, embedding_dim)

# Get outputs
# embeddings, hidden_states, attention_weights, predictions = model(input_ids)

# Visualization functions
# def plot_matrix(matrix, title, words, is_attention=False):
#     plt.figure(figsize=(10, 8))
#     if is_attention:
#         sns.heatmap(matrix, xticklabels=words, yticklabels=words, annot=True, fmt='.2f')
#     else:
#         # For embeddings and hidden states, we show word × embedding_dim
#         sns.heatmap(matrix, xticklabels=range(embedding_dim), 
#                    yticklabels=words, annot=True, fmt='.2f')
#     plt.title(title)
#     plt.show()

# Print results

# print("\nOriginal Embeddings:")
# print(embeddings.detach().tolist())
# plot_matrix(embeddings.detach().tolist(), "Word Embeddings", words)

# print("\nAttention Weights:")
# print(attention_weights.detach().tolist())
# plot_matrix(attention_weights.detach().tolist(), "Attention Weights", words, is_attention=True)

# print("\nHidden States:")
# print(hidden_states.detach().tolist())
# plot_matrix(hidden_states.detach().tolist(), "Hidden States", words)


def create_masked_input(input_ids, mask_token_id=vocab_size):
    masked_input = input_ids.clone()  # Copy of [0, 1, 2, 3, 4]
    labels = input_ids.clone()        # Copy of [0, 1, 2, 3, 4]
    
    # Create random mask (True/False for each position)
    mask = torch.rand(len(input_ids)) < 0.15  # 15% chance of True
    print("\nMask (True means will be masked):", mask.tolist())
    
    # Replace masked positions with mask_token_id (which is 5)
    masked_input[mask] = mask_token_id
    print("Masked input:", masked_input.tolist())
    
    # Set non-masked positions in labels to -100
    labels[~mask] = -100
    print("Labels:", labels.tolist())
    
    return masked_input, labels


def train_model(model, input_ids):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Phase 1: Train on full sequence
    print("\nPhase 1: Training on full sequence")
    for epoch in range(200):  # First 100 epochs on full sequence
        # Forward pass with full sequence
        logits = model(input_ids)
        
        # All positions are training targets
        loss = criterion(logits.view(-1, vocab_size), input_ids)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"\nEpoch {epoch}, Loss: {loss.item():.4f}")
            probs = torch.softmax(logits, dim=-1)
            pred_words = [words[idx] for idx in torch.argmax(logits, dim=-1)]
            print("Predictions:", pred_words)
            print("True words:", [words[i] for i in input_ids.tolist()])
    
    # Phase 2: Train with masking
    print("\nPhase 2: Training with masking")
    for epoch in range(500):  # Remaining epochs with masking
        masked_input, labels = create_masked_input(input_ids)
        
        logits = model(masked_input)
        loss = criterion(logits.view(-1, vocab_size), labels)

        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"\nEpoch {epoch}, Loss: {loss.item():.4f}")
            
            masked_sequence = ['[MASK]' if i == vocab_size else words[i] 
                             for i in masked_input.tolist()]
            print("Masked sequence:", masked_sequence)
            
            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            pred_words = [words[idx] for idx in torch.argmax(logits, dim=-1)]
            
            # For each masked position, show prediction
            for i, (mask, label) in enumerate(zip(masked_input, labels)):
                if mask == vocab_size:  # if position was masked
                    print(f"\nPosition {i}:")
                    print(f"True word: {words[label]}")
                    print(f"Predicted: {pred_words[i]}")
                    print("Probabilities:")
                    # Show probability for each possible word
                    word_probs = probs[i].tolist()
                    for word, prob in zip(words, word_probs):
                        print(f"{word}: {prob:.3f}")

# Train the model
model = SimpleTransformer(vocab_size, embedding_dim)
train_model(model, input_ids)