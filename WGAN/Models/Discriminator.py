import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from config import cfg

class Discriminator16(nn.Module):

    def __init__(self):
        super(Discriminator16, self).__init__()
        

        self.conv1=nn.Sequential(
            nn.Conv3d(4,64, kernel_size=4, stride=1,padding='same'),
            nn.LeakyReLU(0.2)
        )
        self.conv2=nn.Sequential(
            nn.Conv3d(64,128 , kernel_size=4, stride=2,padding=3),
            nn.LeakyReLU(0.2)
        )
        self.conv3=nn.Sequential(
            nn.Conv3d(128,256 , kernel_size=4, stride=1,padding='same'),
            nn.LeakyReLU(0.2)
        )
        self.conv4=nn.Sequential(
            nn.Conv3d(256,512 , kernel_size=4, stride=2,padding=3),
            nn.LeakyReLU(0.2)
        )
        self.conv5=nn.Sequential(
            nn.Conv3d(512,256 , kernel_size=2, stride=1,padding='same'),
            nn.LeakyReLU(0.2)
        )

        embedding_dim = 256

        self.emb_out = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.concat=nn.Sequential(
            nn.Linear(88064, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )
        

    def forward(self, x,emb):
        x = self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x = x.view(128,-1)

        emb=self.emb_out(emb)
        
        out=torch.cat((x,emb),1)

        out=self.concat(out)

        sig=torch.sigmoid(out)        

        return {'sigmoid_output': sig, 'logits': out}
    
class Discriminator32(nn.Module):

    def __init__(self):
        super(Discriminator32, self).__init__()
        

        self.conv1=nn.Sequential(
            nn.Conv3d(4,64, kernel_size=4, stride=1,padding='same'),
            nn.LeakyReLU(0.2)
        )
        self.conv2=nn.Sequential(
            nn.Conv3d(64,128 , kernel_size=4, stride=2,padding=2),
            nn.LeakyReLU(0.2)
        )
        self.conv3=nn.Sequential(
            nn.Conv3d(128,256 , kernel_size=4, stride=2,padding=2),
            nn.LeakyReLU(0.2)
        )
        self.conv4=nn.Sequential(
            nn.Conv3d(256,512 , kernel_size=4, stride=2,padding=2),
            nn.LeakyReLU(0.2)
        )
        self.conv5=nn.Sequential(
            nn.Conv3d(512,256 , kernel_size=2, stride=2,padding=2),
            nn.LeakyReLU(0.2)
        )

        embedding_dim = 256

        self.emb_out = nn.Sequential(
            nn.Linear(128+cfg.GAN_NOISE_SIZE , embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.concat=nn.Sequential(
            nn.Linear(16640, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )
        

    def forward(self, x,emb):
        x = self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x = x.view(cfg.GAN_BATCH_SIZE,-1)

        emb=self.emb_out(emb)
        
        out=torch.cat((x,emb),1)

        out=self.concat(out)

        sig=torch.sigmoid(out)        

        return {'sigmoid_output': sig, 'logits': out}