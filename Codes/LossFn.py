import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

class UncertaintyLoss(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.params = params
        sigma = torch.randn(2)
        self.sigma = nn.Parameter(sigma)
        self.mse = nn.MSELoss()

    def forward(self,Preds,Trues):
        losses = 0
        self.mses = self.mse(Preds,Trues)
        losses +=  (0.5/(self.sigma[0]**2)*self.mses + torch.log(self.sigma[0]**2+1))  
        if self.params['Robust_loss']:
            corr,_ = pearsonr(Preds.flatten().cpu().detach().numpy(),Trues.flatten().cpu().detach().numpy())
            self.pearsons = abs(1-corr)*1000
            losses +=  (0.5/(self.sigma[1]**2)*self.pearsons + torch.log(self.sigma[1]**2+1))  
        return losses
    
    def getLossComponent(self):
        if self.params['Robust_loss']:
            return self.mses, self.pearsons
        else:
            return self.mses
            



