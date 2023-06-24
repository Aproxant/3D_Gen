import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RoundTripLoss(nn.Module):
    def __init__(self, weight=1.):
        super(RoundTripLoss, self).__init__()
        self.weight = weight

    def forward(self, a, b, targets):

        # similarity
        sim = a.matmul(b.transpose(1, 0).contiguous())
        # walk
        a2b = F.softmax(sim, dim=1)
        b2a = F.softmax(sim.transpose(1, 0), dim=1)
        # build inputs
        inputs = a2b.matmul(b2a)

              
        return self.weight*nn.CrossEntropyLoss()(1e-8+inputs, targets) 
        #targets.mul(torch.log(1e-8 + inputs)).sum(1).mean()#dodac epsilon
        # return self.weight * self.loss(torch.log(1e-8 + inputs), targets)


class AssociationLoss(nn.Module):
    def __init__(self, weight=1.):
        super(AssociationLoss, self).__init__()
        self.weight = weight
        
    def forward(self, a, b):
        # similarity
        sim = a.matmul(b.transpose(1, 0).contiguous())
        # visit
        a2b = F.softmax(sim, dim=1)
        visit_probability  = a2b.mean(0, keepdim=True)
        t_nb=a2b.shape[1]
        # build targets
        targets = torch.full((1,t_nb),1./float(t_nb)).to(device)

        return self.weight * nn.CrossEntropyLoss()(1e-8 + visit_probability,targets) #targets.mul(torch.log(1e-8 + visit_probability)).sum(1).mean()#self.weight*self.loss(1e-8 +visit_probability,targets)


class SmoothedMetricLoss(nn.Module):
    def __init__(self, margin=1.):
        super(SmoothedMetricLoss, self).__init__()
        self.margin = margin
    
    def _get_distance(self, X):
        # similarity
        
        D = X.matmul(X.transpose(1, 0))

        D /= 128.
        #print(D)
        # get exponential distance
        Dexpm = torch.exp(self.margin + D)

        return D, Dexpm
        

    def forward(self, inputs):
        '''
        instance-level metric learning loss, assuming all inputs are from different catagories
        labels are not needed
        param:
            inputs: composed embedding matrix with assumption that two consecutive data pairs are from the same class
                    note that no other data pairs in this batch can be from the same class
        
        return:
            instance-level metric_loss: see https://arxiv.org/pdf/1803.08495.pdf Sec.4.2
        '''

        batch_size = inputs.size(0)
        
        # get distance
        D, Dexpm = self._get_distance(inputs)

        global_comp = [0.] * (batch_size // 2)
        for pos_id in range(batch_size // 2):
            pos_i = pos_id * 2
            pos_j = pos_id * 2 + 1
            pos_pair = (pos_i, pos_j)

            neg_i = [pos_i * batch_size + k * 2 + 1 for k in range(batch_size // 2) if k != pos_j]
            neg_j = [pos_j * batch_size + l * 2 for l in range(batch_size // 2) if l != pos_i]

            neg_ik = Dexpm.take(torch.LongTensor(neg_i).to(device)).sum()
            neg_jl = Dexpm.take(torch.LongTensor(neg_j).to(device)).sum()
            Dissim = neg_ik + neg_jl

            J_ij = torch.log(1e-8 + Dissim).to(device) - D[pos_pair]


            max_ij = torch.max(J_ij, torch.zeros(J_ij.size()).to(device)).pow(2)            
            global_comp[pos_id] = max_ij.unsqueeze(0)
        
        # accumulate
        outputs = torch.cat(global_comp).sum().div(batch_size)

        return outputs

        """
        J_all=[]

        for pos_id in range(batch_size // 2):
            i = pos_id * 2
            j = i + 1

            ind_rest = np.hstack([np.arange(0, pos_id * 2),np.arange(pos_id * 2 + 2, batch_size)])
            inds = [[i, k] for k in ind_rest]
            inds.extend([[j, l] for l in ind_rest])

            inds=torch.LongTensor(inds).to(device)
            pos_h = inds[:, 0]
            pos_w = inds[:, 1]
          

            
            J_ij = torch.log(1e-8 + Dexpm[pos_h,pos_w].sum())-D[i,j]
            J_all.append(J_ij)


        J_all=torch.FloatTensor(J_all).to(device)

            
        loss = torch.max(J_all, torch.zeros(J_all.size()).to(device)).pow(2).sum().mean().div(2.0)            

        return loss
        """