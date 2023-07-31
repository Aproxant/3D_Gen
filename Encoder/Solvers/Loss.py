import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg


class RoundTripLoss(nn.Module):
    def __init__(self,device='cpu'):
        super(RoundTripLoss, self).__init__()

        self.device=device
    def forward(self, T, S, targets):
        
        # similarity matrix
        M= T.matmul(S.transpose(1, 0).contiguous())

        #probability
        T2S = F.softmax(M, dim=1)
        S2T = F.softmax(M.transpose(1, 0), dim=1)

        # build inputs
        p_TST = T2S.matmul(S2T)

              
        return nn.CrossEntropyLoss()(cfg.EPS+p_TST, targets) 
    

class AssociationLoss(nn.Module):
    def __init__(self ,device='cpu'):
        super(AssociationLoss, self).__init__()
        self.device=device
    def forward(self, T, S):
        # similarity matrix
        M = T.matmul(S.transpose(1, 0).contiguous())

        #probability
        T2S = F.softmax(M, dim=1)

        visit_probability  = T2S.mean(0, keepdim=True)
        S_size=T2S.shape[1]

        # build targets
        targets = torch.full((1,S_size),1./float(S_size)).to(self.device)

        return nn.CrossEntropyLoss()(cfg.EPS + visit_probability,targets) 

class InstanceMetricLoss(nn.Module):
    def __init__(self, margin=1.):
        super(InstanceMetricLoss, self).__init__()
        self.margin = margin
    
    def _get_distance(self, inputs):

        D = inputs.matmul(inputs.transpose(1, 0))

        D /= 128.

        Dexpm = torch.exp(self.margin + D)


        return D, Dexpm
        

    def forward(self, inputs):

        batch_size = inputs.size(0)
        
        # get distance
        D, Dexpm = self._get_distance(inputs)

        # compute pair-wise loss
        global_comp = [0.] * (batch_size // 2)
        for pos_id in range(batch_size // 2):
            pos_i = pos_id * 2
            pos_j = pos_i + 1
            pos_pair = (pos_i, pos_j)

            neg_i = [pos_i * batch_size + k * 2 + 1 for k in range(batch_size // 2) if k != pos_j]
            neg_j = [pos_j * batch_size + l * 2 for l in range(batch_size // 2) if l != pos_i]

            neg_ik = Dexpm.take(torch.LongTensor(neg_i)).sum()
            neg_jl = Dexpm.take(torch.LongTensor(neg_j)).sum()
            Dissim = neg_ik + neg_jl


            J_ij = torch.log(1e-8 + Dissim) - D[pos_pair]


            max_ij = torch.max(J_ij, torch.zeros(J_ij.size())).pow(2)            
            global_comp[pos_id] = max_ij.unsqueeze(0)
        
        # accumulate
        outputs = torch.cat(global_comp).sum().div(batch_size)

        return outputs
    

