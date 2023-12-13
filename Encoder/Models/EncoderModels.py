import torch
import torch.nn as nn
from config import cfg
import torch.nn.functional as F
import math
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self,dict_size):
        super(TextEncoder, self).__init__()

        #W1 = torch.FloatTensor(np.random.uniform(-1,1,size=(dict_size+1,128)))
        self.embedded = torch.nn.Embedding(dict_size+1,128, padding_idx=0)
        #self.embedded.weight.requires_grad = False

        self.conv_128 = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.bn_128 = nn.BatchNorm1d(128)
        self.conv_256 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        self.bn_256 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(256, 256, batch_first=True)

        self.outputs = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            #nn.Sigmoid()
        )

    def forward(self, inputs):

        embedded = self.embedded(inputs) 
        conved = self.conv_128(embedded.transpose(2, 1).contiguous()) 
        conved = self.bn_128(conved)
        conved = self.conv_256(conved) 

        conved = self.bn_256(conved).transpose(2, 1).contiguous()
        _, (hidden_state, _)= self.lstm(conved, None)
        outputs = self.outputs(hidden_state[-1])
        #normalized_text_encoding = F.normalize(outputs, p=2, dim=1)
        return outputs
    


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalSelfAttention, self).__init__()
        # basic settings
        self.hidden_size = hidden_size
        # MLP
        self.comp_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, 1, bias=False)
        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)

    def forward(self, h):
        h_comp = self.comp_h(h) # (batch_size, seq_size, hidden_size)
        outputs = F.softmax(self.output_layer(h_comp), dim=1) # (batch_size, seq_size, 1)

        return outputs

class TextEncoderWithATTN(nn.Module):
    def __init__(self, dict_size, embed_size=128, heads=8):
        super(TextEncoderWithATTN, self).__init__()

        W1 = torch.FloatTensor(np.random.uniform(-1, 1, size=(dict_size, embed_size)))
        self.embedded = torch.nn.Embedding(
            dict_size, embed_size, _weight=W1, padding_idx=0
        )
        # self.embedded.weight.requires_grad = False

        self.conv_128 = nn.Sequential(
            nn.Conv1d(embed_size, embed_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_size, embed_size, 3, padding=1),
            nn.ReLU(),
        )
        self.bn_128 = nn.BatchNorm1d(embed_size)
        self.conv_256 = nn.Sequential(
            nn.Conv1d(embed_size, embed_size * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_size * 2, embed_size * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.bn_256 = nn.BatchNorm1d(embed_size * 2)
        
        self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.attention_temporal = TemporalSelfAttention(256)
        self.outputs = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            #nn.Sigmoid()
        )
    def _init_hidden(self, text_feat):
        states = (
            torch.zeros(1, text_feat.size(0), 256).to(cfg.DEVICE),
            torch.zeros(1, text_feat.size(0), 256).to(cfg.DEVICE)
        )

        return states

    def attend(self, text_feat, states):
        temporal_weights = self.attention_temporal(text_feat)
        temporal_attended = torch.sum(text_feat * temporal_weights, dim=1) # (batch_size, hidden_size)

        return temporal_attended, temporal_weights

    def forward(self, inputs):
        embedded = self.embedded(inputs) 

        conved = self.conv_128(embedded.transpose(2, 1).contiguous()) 
        conved = self.bn_128(conved)
        conved = self.conv_256(conved) 

        conved = self.bn_256(conved).transpose(2, 1).contiguous()
        states = self._init_hidden(conved) 
        text_feat, _ = self.lstm(conved, states)

        temporal_attended, _ = self.attend(text_feat, states)
        text_outputs = self.outputs(temporal_attended)


        return text_outputs


class TransferLearningALBERT(nn.Module):
    def __init__(self):
        super(TransferLearningALBERT, self).__init__()
        self.transferLearning = AlbertModel.from_pretrained('albert-base-v2') 
        for param in self.transferLearning.parameters():
            param.requires_grad = False
        self.outputs = nn.Sequential(
            nn.Linear(self.transferLearning.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            #nn.Sigmoid()
        ) 


    def forward(self,x):
        x=self.transferLearning(**x)
        x=self.outputs(x[1])
        normalized_text_encoding = F.normalize(x, p=2, dim=1)

        return normalized_text_encoding