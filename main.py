import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, d_model, d_proj, vocab_size, pad_id):
        super().__init__()
        self.d_model = d_model
        self.d_proj = d_proj
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model)
        
    def forward(self,input_ids):
        # Creating mask
        mask_bool = 