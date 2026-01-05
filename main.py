import torch
import torch.nn as nn
from data import training_corpus

class Encoder_f(nn.Module):
    def __init__(self,d_input,d_model, h):
        super().__init__()
        self.ln1 = nn.Linear(d_input,d_model)
        self.ln2 = nn.Linear(d_model, h)

    def forward(self,x):
        #Applying MLP
        #Applying first FFLayer
        output = self.ln1(x)
        #ReLu
        output = nn.ReLU(x)
        #Applying second FFLayer
        output = self.ln2(x)
        #Returning model output
        return output

class ProjectionHead_g(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        ...

class ContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        ...