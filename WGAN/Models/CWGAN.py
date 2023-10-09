import torch
import torch.nn as nn
import torch.nn.functional as F
from Discriminator import Discriminator32
from Generator import Generator32

class CWGAN(nn.Module):
    def __init__(self):
        self.generator_class = Generator32()
        self.critic_class = Discriminator32()
        super(CWGAN, self).__init__()

    
    def forward(self,shape,emb):
        gen_out=self.generator_class(emb)
        critic_out=self.critic_class(shape,emb)

        return gen_out, critic_out

        
