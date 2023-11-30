import torch
import torch.nn as nn
import numpy as np
from config import cfg
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F


class InstanceMetricLoss(nn.Module):
    def __init__(self,device='cpu'):
        super(InstanceMetricLoss, self).__init__()
        self.margin = cfg.METRIC_MARGIN
        self.device=device

    def forward(self, inputs,t):
        
        batch_size = inputs.size(0)
        
        Xe = torch.unsqueeze(inputs, 1)
        D = torch.sum(Xe * Xe.permute(1, 0, 2), dim=2)
        D /= 128.0
        
        expmD = torch.exp(self.margin + D)
        
        J_all = []

        for pair_ind in range(batch_size // 2):
            i = pair_ind * 2
            j = i + 1
            ind_rest = np.hstack([np.arange(0, pair_ind * 2),
                          np.arange(pair_ind * 2 + 2, batch_size)])

            inds = np.array([[i, k] for k in ind_rest] + [[j, l] for l in ind_rest])
            J_ij = torch.log(torch.sum(expmD[inds[:, 0], inds[:, 1]])) - D[i, j]

            J_all.append(J_ij)

        # Convert J_all to a PyTorch tensor
        J_all = torch.stack(J_all)

        # Compute the loss
        loss = torch.div(torch.mean(torch.square(torch.clamp(J_all, 0.0))), 2.0)
        """
        D = inputs.matmul(inputs.transpose(1, 0))
        D /= 128.0
        expmD = torch.exp(self.margin + D)

        global_comp = [0.] * (batch_size // 2)
        for pos_id in range(batch_size // 2):
            pos_i = pos_id * 2
            pos_j = pos_id * 2 + 1
            pos_pair = (pos_i, pos_j)

            neg_i = [pos_i * batch_size + k * 2 + 1 for k in range(batch_size // 2) if k != pos_j]
            neg_j = [pos_j * batch_size + l * 2 for l in range(batch_size // 2) if l != pos_i]


            neg_ik = expmD.take(torch.LongTensor(neg_i).to(self.device)).sum()
            neg_jl = expmD.take(torch.LongTensor(neg_j).to(self.device)).sum()
            Dissim = neg_ik + neg_jl


            J_ij = torch.log(1e-8 + Dissim).to(self.device) - D[pos_pair]


            max_ij = torch.max(J_ij, torch.zeros(J_ij.size()).to(self.device)).pow(2)            
            global_comp[pos_id] = max_ij.unsqueeze(0)
        
        # accumulate
        loss = torch.cat(global_comp).sum().div(batch_size)
        """
        return loss
    



class TripletLoss(nn.Module):
    def __init__(self,device='cpu'):
        super(TripletLoss, self).__init__()
        self.margin = cfg.METRIC_MARGIN_TRIPLET
        self.device=device
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2+cfg.EPS).pow(2).sum(1).sqrt()
    def calc_cos(self, x1, x2):
        normMul=torch.norm(x1,dim=1)*torch.norm(x2,dim=1)
        return ((x1*x2).sum(1))/normMul
    
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
        losses = torch.relu(distance_positive - distance_negative + self.margin).sum(0)

        return losses/batch_size
    
class customSimilarityLoss(nn.Module):
    def __init__(self,device='cpu'):
        super(customSimilarityLoss, self).__init__()

    def forward(self, embeddings,t_labels):

        cosine_sim_matrix = torch.from_numpy(cosine_similarity(embeddings.detach().numpy(), embeddings.detach().numpy())).float()
        normalized_cosine_sim_matrix = (cosine_sim_matrix + 1) / 2
        binary_labels=t_labels.unsqueeze(1)*t_labels.unsqueeze(0)
        normalized_cosine_sim_matrix = normalized_cosine_sim_matrix.view(-1)
        print(torch.max(normalized_cosine_sim_matrix))
        print(torch.min(normalized_cosine_sim_matrix))
        binary_labels_flat = binary_labels.view(-1).float()
        print(torch.min(binary_labels_flat))
        print(torch.max(binary_labels_flat))
        bce_loss = F.binary_cross_entropy(normalized_cosine_sim_matrix, binary_labels_flat)

        return bce_loss

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

        n_pairs, n_negatives = self.get_n_pairs(t_labels)
        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:,1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_loss(anchors, positives, negatives) + self.l2_reg * self.l2_loss(anchors, positives)
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

        for i in range(len(anchor_idx)):
            n_pairs.append([anchor_idx[i], positive_idx[i]])
    
        
        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i,1], n_pairs[i+1:,1]])
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
        anchors = torch.unsqueeze(anchors, dim=1)  
        positives = torch.unsqueeze(positives, dim=1)  

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2)) 
        x = torch.sum(torch.exp(x), 2) 
        loss = torch.mean(torch.log(1+x))
        return loss
        """
        pos_sum=(anchors @ positives.transpose(0,1)).diagonal()
        transposed_negatives = negatives.transpose(1, 2)

        result = torch.zeros((anchors.shape[0], negatives.shape[1]))

        # Perform the element-wise multiplications and summation
        for i in range(anchors.shape[0]):
            result[i] = torch.matmul(anchors[i],transposed_negatives[i])-pos_sum[i]


        x = torch.sum(torch.exp(result), 1)
        loss = torch.mean(torch.log(1+x))
        return loss
        """

    def l2_loss(self,anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]




