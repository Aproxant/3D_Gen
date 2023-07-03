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


class SmoothedMetricLoss(nn.Module):
    def __init__(self,device='cpu'):
        super(SmoothedMetricLoss, self).__init__()
        self.device=device

    def forward(self, X):

        batch_size = X.size(0)

        Xe=X.unsqueeze(1)

        D=torch.mul(Xe.expand(Xe.shape[0],Xe.shape[0],Xe.shape[2]),Xe.permute(1,0,2).expand(Xe.shape[0],Xe.shape[0],Xe.shape[2]))
        D=D.sum(2,keepdim=False).div(cfg.EMBEDDING_DIM)

        expmD=torch.exp(D+cfg.METRIC_MARGIN)

        J_all=[]
        for pos_id in range(batch_size // 2):
            pos_i = pos_id * 2
            pos_j = pos_i+ 1

            ind_rest = np.hstack([np.arange(0, pos_id * 2),
                                  np.arange(pos_id * 2 + 2, batch_size)])
            
            inds = [[pos_i, k] for k in ind_rest]
            inds.extend([[pos_j, l] for l in ind_rest])
            J_ij = torch.log(cfg.EPS+expmD[tuple(zip(*inds))].sum()) - D[pos_i, pos_j]
            J_all.append(J_ij)

        J_all = torch.tensor(J_all)

        loss = torch.max(J_all, torch.zeros(J_all.size()).to(self.device)).pow(2).mean().div(2.0)            
        return loss