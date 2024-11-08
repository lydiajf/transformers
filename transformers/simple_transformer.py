import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
    

sentence = "hello my name is lydia"
words = sentence.split()
vocab = {word: idx for idx, word in enumerate(words)}
vocab_size = len(vocab)

embedding_dim = 8

class Magic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.w_Q = nn.Linear(embedding_dim, embedding_dim)
        self.w_K = nn.Linear(embedding_dim, embedding_dim)
        self.w_V = nn.Linear(embedding_dim, embedding_dim)

    def forward(self,x):
        Q = self.w_Q(x)
        K = self.w_K(x)
        V = self.w_V(x)
        attn = torch.matmul(Q, K.transpose(0,1))
        out = torch.matmul(attn, V)
        return out


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length=5, num_layers=4):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size + 1, embedding_dim)  # +1 for mask token
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.magic = nn.ModuleList([Magic(embedding_dim) for _ in range(num_layers)])

        self.prediction_head = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        # Get word embeddings
        word_embeddings = self.word_embedding(x)
        
        # Add position embeddings
        positions = torch.arange(len(x)).to(x.device)
        pos_embeddings = self.position_embedding(positions)
        
        # Combine word and position embeddings
        x = word_embeddings + pos_embeddings.unsqueeze(0)
         # Pass through Magic layers
        for magic_layer in self.magic:
            x = magic_layer(x)
        
        # Predict
        logits = self.prediction_head(x)
        
        return logits

# Create input tensor
input_ids = torch.tensor([vocab[word] for word in words])
print("Original input_ids:", input_ids.tolist())


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