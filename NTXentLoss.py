import torch
import torch.nn as nn

class NTXent(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, z1, z2):
        self.B = z1.shape[0]
        z = torch.cat([self.l2norm(z1),self.l2norm(z2)],dim=0)
        S, positives = self.similarityMatrix(z)
        S = self.masking(S)
        loss = self.lossCalc(S,positives)
        return loss

    def l2norm(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=1)
    
    def similarityMatrix(self,z):
        S = z @ z.T
        rows = torch.arange(2*self.B, device = z.device)
        cols = (rows + self.B) % 2*self.B
        positives = S[rows,cols]
        return S, positives
    
    def masking(self, S):
        mask = torch.eye(2*self.B, dtype=torch.bool).to(S.device)
        S = S.masked_fill_(mask=mask, value=float('-inf'))
        return S

    def lossCalc(self,S,positives):
        denominator =  torch.sum(torch.exp(S/self.tau),1)
        numerator = torch.exp(positives/self.tau)
        big_L = -1 * torch.sum(torch.log(numerator/denominator))
        return(big_L/(2*self.B))
