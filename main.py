import torch
import torch.nn as nn
import torch.nn.functional as F
from NTXentLoss import NTXent
from data import Dataset, collate_fn, train_text, valid_text, test_text, vocab
from torchtext.data import get_tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        z = self.ln2(F.relu(self.ln1(h))) #Applying Linear Layer -> Non-linearity -> Second Layer -> z (B,d_proj)

        return h, z

def main():
    # Setting up Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # Flag if mps not available
    if device == 'cpu':
        raise Exception("GPU not available. Please run on a Mac with MPS support.")

    # Instantiate Dataset
    dataset = Dataset(p_dropout = 0.2, data = train_text, tokenizer = get_tokenizer("basic_english"), vocab = vocab)
    # Instantiate DataLoader
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)
    # Instantiate Encoder
    encoder = Encoder(d_model = 64, d_proj = 16, vocab_size = vocab.__len__(), pad_id = vocab["<unk>"]).to(device)
    # Instantiate Loss Function
    NTXentLoss = NTXent(tau=0.1)
    # Instantiate Optimiser
    optimiser = torch.optim.Adam(params=encoder.parameters())
    # Running training loop
    trained_encoder, losses = training_loop(dataloader,encoder,NTXentLoss,optimiser,epoch_number = 1,device='mps')

    # Quick Smoke test for exploding or NaN gradients
    print('Encoder Losses Smoke Test:')
    print(losses[0:10])

    # Saving Encoder
    if device == 'mps':
            torch.save(trained_encoder.state_dict(), './')

    # Indexing eval dataset from training (same dataset used as no performance metrics being evaluated)
    eval_text = train_text[:500]
    # Instantiating eval Dataset
    eval_dataset = Dataset(p_dropout = 0.2, data = eval_text, tokenizer = get_tokenizer("basic_english"), vocab = vocab)
    

    # Visual Eval
    model_eval(eval_dataloader, trained_encoder)


def training_loop(dataloader,encoder,NTXentLoss,optimiser,epoch_number,device):
    
    for epoch in range(epoch_number):
        encoder.train()
        losses = []
        loop = tqdm(dataloader, leave = True)
        for batch in loop:
            views1, views2 = batch[0], batch[1] # Batched views Shape:(B, padded_length)

            # Moving views1 and views2 to device
            views1, views2 = views1.to(device), views2.to(device)
            # Passing the views to the Encoder
                # Input Shape: (B,L) 
                # Output Shape: z: (B, d_proj), h:(B, d_Encoder)

            h1, z1 = encoder(views1)
            h2, z2 = encoder(views2)

            # Passing NTXentLoss funct z1 and z2 fo loss calc
                # Input Shape z: (B, d_proj), Output Shape: (1)
            loss = NTXentLoss(z1, z2)

            # Saving loss for current batch for loss curve 
            losses.append(loss.item())

            # Zeroing gradients from previous loop
            optimiser.zero_grad()
            # Backprop Loss
            loss.backward()
            # Updating param values
            optimiser.step() 

            loop.set_description(f"Epoch [{epoch+1}/{epoch_number}]")

    print("Training Complete.")

    return encoder, losses

def model_eval(eval_dataloader,trained_encoder):
    # Setting encoder to eval mode
    trained_encoder.eval()
    # 


if __name__ == '__main__':
    main()



"""

NOTES:
* Only tensors and modules live on devices, therfore cannot/should not use it on dataset and dataloader
* When loss.backward() is run, this calculates the gradients of all the params which are then stored in param.grad() for each paramter. 
    Optimiser.step() then uses the gradients to update the params?
# The data/batch instances should sit on a device but this should be in the training loop not when the dataloader/set is instantiated.
# Loss calculators and optimisers also dont sit on a device.

"""