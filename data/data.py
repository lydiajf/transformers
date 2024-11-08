from datasets import load_dataset
import numpy as np
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers.tokenizer import Tokenizer

def load_and_prepare_data(max_examples=1000):
    """Load dataset and prepare text"""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    
    # Concatenate texts (limited for testing)
    train_text = " ".join(ds["train"]["text"][:max_examples])
    validation_text = " ".join(ds["validation"]["text"][:max_examples//10])
    test_text = " ".join(ds["test"]["text"][:max_examples//10])
    
    return train_text, validation_text, test_text

def create_segments(tokens, max_length=150, pad_id=0):
    """Create segments of fixed length from token list"""
    segments = []
    for i in range(0, len(tokens), max_length):
        segment = tokens[i:i + max_length]
        if len(segment) < max_length:
            segment = segment + [pad_id] * (max_length - len(segment))
        segments.append(segment)
    return segments

def create_masked_segments(segments, tokenizer, mask_prob=0.15):
    """Create masked versions of segments"""
    masked_segments = []
    labels = []
    
    for segment in segments:
        masked_segment = segment.copy()
        label = [-100] * len(segment)  # -100 is ignore index
        
        # Only consider non-padding tokens for masking
        mask_candidates = [i for i, token in enumerate(segment) 
                         if token != tokenizer.pad_id()]
        
        # Select tokens to mask
        n_to_mask = int(len(mask_candidates) * mask_prob)
        to_mask = random.sample(mask_candidates, n_to_mask)
        
        for idx in to_mask:
            label[idx] = segment[idx]  # Save original token
            masked_segment[idx] = tokenizer.sp.piece_to_id('[UNK]')
        
        masked_segments.append(masked_segment)
        labels.append(label)
    
    return masked_segments, labels

def main():
    # Initialize tokenizer
    tokenizer = Tokenizer()
    
    # Load and prepare data
    train_text, val_text, test_text = load_and_prepare_data()
    
    # Save training text for tokenizer training
    with open("train_text.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    
    # Train tokenizer
    tokenizer.train("train_text.txt")
    
    # Tokenize texts
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    test_tokens = tokenizer.encode(test_text)
    
    # Create segments
    train_segments = create_segments(train_tokens, pad_id=tokenizer.pad_id())
    val_segments = create_segments(val_tokens, pad_id=tokenizer.pad_id())
    test_segments = create_segments(test_tokens, pad_id=tokenizer.pad_id())
    
    # Create masked versions
    train_masked, train_labels = create_masked_segments(train_segments, tokenizer)
    val_masked, val_labels = create_masked_segments(val_segments, tokenizer)
    test_masked, test_labels = create_masked_segments(test_segments, tokenizer)
    
    # Convert to numpy arrays and save
    np.save('train_segments.npy', np.array(train_segments))
    np.save('val_segments.npy', np.array(val_segments))
    np.save('test_segments.npy', np.array(test_segments))
    
    np.save('train_masked.npy', np.array(train_masked))
    np.save('train_labels.npy', np.array(train_labels))
    np.save('val_masked.npy', np.array(val_masked))
    np.save('val_labels.npy', np.array(val_labels))
    np.save('test_masked.npy', np.array(test_masked))
    np.save('test_labels.npy', np.array(test_labels))
    
    # Save vocabulary size
    np.save('vocab_size.npy', tokenizer.get_vocab_size())
    
    print("Data processing complete!")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Number of training segments: {len(train_segments)}")

if __name__ == "__main__":
    main()