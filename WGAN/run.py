import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import trimesh
from functools import partial
import ssl
import pickle
from config import cfg
import torch.optim as optim

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

from DataLoader.GANloader import GANLoader
from DataLoader.GanDataGen import GANDataGenerator
from Models.Generator import Generator32,Generator32_Small
from Models.Discriminator import Discriminator32,Discriminator32_Small

from Solvers.SolverGAN import SolverGAN
device=cfg.DEVICE
#print(device)
PYTORCH_ENABLE_MPS_FALLBACK=1
#for mac os fix 
ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

if __name__=='__main__':
    GAN_Data=GANDataGenerator()
    generator=Generator32().to(cfg.DEVICE)
    critic=Discriminator32().to(cfg.DEVICE)
    d_optimizer = optim.RMSprop(critic.parameters(), lr=cfg.GAN_LR, weight_decay=cfg.GAN_WEIGHT_DECAY)
    g_optimizer = optim.RMSprop(generator.parameters(), lr=cfg.GAN_LR, weight_decay=cfg.GAN_WEIGHT_DECAY)


    optimizer={'disc':d_optimizer,
           'gen':g_optimizer}

    history=SolverGAN(GAN_Data,generator,critic,optimizer)
    history.train(10)