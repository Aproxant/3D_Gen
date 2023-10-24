from torch.utils.data import DataLoader
import torch
import ssl
from Solvers import SolverEmbedding,Loss
from scipy.stats import norm
import os
from Models.EncoderModels import TextEncoderWithATTN,TransferLearningALBERT,TextEncoder
from config import cfg
from dataEmbedding.dataEmbedding import Read_Load_BuildBatch
from dataEmbedding.dataEmbeddingLoader import GenerateDataLoader,check_dataset,collate_embedding
from dataEmbedding.generateEmbedding import build_embeedings_CWGAN
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import pickle
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

device=cfg.DEVICE
#print(device)
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
PYTORCH_ENABLE_MPS_FALLBACK=1
#for mac os fix 
ssl._create_default_https_context = ssl._create_unverified_context


if __name__=='__main__':    
    stanData=Read_Load_BuildBatch(cfg.EMBEDDING_BATCH_SIZE)


    criterion={
        #'metric_main': Loss.InstanceMetricLoss(device=cfg.DEVICE),
        'metric_separator': Loss.TripletLoss(device=cfg.DEVICE),
        'metric_main':Loss.NPairLoss(device=cfg.DEVICE),
        #'metric_separator':Loss.customSimilarityLoss()
        }

    if cfg.EMBEDDING_ALBERT:
        TextModel=TransferLearningALBERT()
        for i in TextModel.parameters():
            i.requires_grad_=False
    else:
        TextModel=TextEncoder(len(stanData.dict_word2idx))
    TextModel=TextModel.to(device)

    optimizer = torch.optim.Adam(TextModel.parameters(), lr=cfg.EMBEDDING_LR, weight_decay=cfg.EMBEDDING_WEIGHT_DC)
    history=SolverEmbedding.Solver(TextModel,stanData,optimizer,criterion,cfg.EMBEDDING_BATCH_SIZE,'online',device)


    history.train(cfg.EMBEDDING_EPOCH_NR,stanData.dict_idx2word)