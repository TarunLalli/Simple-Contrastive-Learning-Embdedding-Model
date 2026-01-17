import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, d_model, d_proj, vocab_size, pad_id):
        super().__init__()
        self.d_model = d_model
        self.d_proj = d_proj
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ln1 = nn.Linear(d_model,d_proj)
        self.ln2 = nn.Linear(d_proj,d_proj)
        
    def forward(self,input_ids): #input_ids: (B,L)
        input_embeddings = self.embed(input_ids) #input_embeddings: (B,L,d_model)
        # Creating mask
        mask = torch.where(input_ids==self.pad_id, 0.0, 1.0).unsqueeze(-1) #mask: (B,L,1)
        # Applying mask to embeddings
        masked_embeddings = input_embeddings * mask #masked_embeddings: (B,L,d_model)
            
        # Mean pooling masked embeddings
        # Masked Embeddings Sum
        pooled_embeddings = torch.sum(masked_embeddings, dim=1) #pooled_embeddings: (B,d_model)
        # Scale by number of true tokens
        true_token_lengths = torch.sum(mask.squeeze(-1), dim=1).unsqueeze(-1) #true_token_lengths: (B,1)
        h = pooled_embeddings/torch.where(true_token_lengths==0, 1,true_token_lengths) #Accounting for Divide by zero errors here by setting empty sequences to length 1.

        #Projection Head Block
        z = self.ln2(F.relu(self.ln1(h))) #Applying Linear Layer -> Non-linearity -> Second Layer -> z

        return h, z
