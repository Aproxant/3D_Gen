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
        

    def forward(self, inputs,t):

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
    

class TripletLoss(nn.Module):
    def __init__(self, margin=1.,device='cpu'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.device=device
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, inputs,t_labels):

        batch_size = inputs.size(0)

        anchor_idx=[k*3 for k in range(batch_size // 3)]
        positive_idx=[k*3+1 for k in range(batch_size // 3)]
        negative_idx=[k*3+2 for k in range(batch_size // 3)]

        anchor = inputs[anchor_idx]
        positive = inputs[positive_idx]
        negative = inputs[negative_idx]


        
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)

        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class NPairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    """

    def __init__(self, l2_reg=0.2,device='cpu'):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.device=device
        
    def forward(self, embeddings,t_labels):
        #labels=[]

        #[labels.extend([i,i]) for i in range(embeddings.shape[0]//2)]
        #labels=np.array(labels)

        n_pairs, n_negatives = self.get_n_pairs(t_labels)
        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:,1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_loss(anchors, positives, negatives) #+ self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    def get_n_pairs(self,labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        n_pairs=[]
        batch_size=labels.size(0)
        anchor_idx=[k*2 for k in range(batch_size // 2)]
        positive_idx=[k*2+1 for k in range(batch_size // 2)]
        #n_pairs = []
        #for label in set(labels):
        #    label_mask = (labels == label)
        #    label_indices = np.where(label_mask)[0]
        #    if len(label_indices) < 2:
        #       continue
        for i in range(len(anchor_idx)):
            n_pairs.append([anchor_idx[i], positive_idx[i]])
    
        
        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.append(n_pairs[:i,1], n_pairs[i+1:,1])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)
        

        return torch.LongTensor(n_pairs).to(self.device), torch.LongTensor(n_negatives).to(self.device)

    def n_pair_loss(self,anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        #anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        #positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)
        
        pos_sum=(anchors @ positives.transpose(0,1)).diagonal()
        transposed_negatives = negatives.transpose(1, 2)

        result = torch.zeros((anchors.shape[0], negatives.shape[1]))

        # Perform the element-wise multiplications and summation
        for i in range(anchors.shape[0]):
            result[i] = torch.matmul(anchors[i],transposed_negatives[i])-pos_sum[i]


        x = torch.sum(torch.exp(result), 1)
        print(torch.log(1+x))
        loss = torch.mean(torch.log(1+x))
        return loss

    def l2_loss(self,anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]




