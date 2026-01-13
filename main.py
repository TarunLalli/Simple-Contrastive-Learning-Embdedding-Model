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
        
    def forward(self,input_ids): #input_ids: (B,L)
        input_embeddings = self.embed(input_ids) #input_embeddings: (B,L,d_model)
        # Creating mask
        mask = torch.where(input_ids==self.pad_id, 0, 1).unsqueeze(-1) #mask: (B,L,1)
        # Applying mask to embeddings
        masked_embeddings = input_embeddings * mask
        # Mean pooling masked embeddings
        pooled_embeddings = torch.sum(input_embeddings, dim=1) #pooled_embeddings: (B,d_model)
        #Scale by number of true tokens
        true_token_lengths = torch.sum(mask.squeeze(-1), dim=1).unsqueeze(-1) #true_token_lengths: (B,1)
        mean_pooled_embeddings = pooled_embeddings/true_token_lengths


