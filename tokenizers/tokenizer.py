import sentencepiece as spm
import os

class Tokenizer:
    def __init__(self, vocab_size=3000, model_prefix='wiki_sp'):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.sp = None
    
    def train(self, text_file):
        """Train the SentencePiece tokenizer"""
        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',  # Using byte-pair encoding
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='[PAD]',
            unk_piece='[UNK]',
            bos_piece='[BOS]',
            eos_piece='[EOS]'
        )
        
        # Load the trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{self.model_prefix}.model')
    
    def load(self):
        """Load an existing tokenizer"""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{self.model_prefix}.model')
    
    def encode(self, text):
        """Encode text to token ids"""
        return self.sp.encode_as_ids(text)
    
    def decode(self, ids):
        """Decode token ids to text"""
        return self.sp.decode_ids(ids.tolist() if hasattr(ids, 'tolist') else ids)
    
    def get_vocab_size(self):
        """Get the vocabulary size"""
        return self.sp.get_piece_size()
    
    def get_words(self):
        """Get list of all tokens in vocabulary"""
        return [self.sp.id_to_piece(i) for i in range(self.get_vocab_size())]
    
    def pad_id(self):
        """Get the ID of the padding token"""
        return self.sp.piece_to_id('[PAD]')
