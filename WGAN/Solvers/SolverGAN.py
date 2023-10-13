
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
import torch
import numpy as np
from config import cfg
import os
from tqdm.notebook import tqdm as tqdm_nb
from torch.utils.data import DataLoader
from DataLoader.GANloader import GANLoader,check_dataset,collate_embedding
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import random
import time


class SolverGAN():
    def __init__(self,data_class, generator,discriminator, optimizer):
        self.data_class=data_class
        self.generator=generator
        self.discriminator=discriminator
        self.batch_size = cfg.GAN_BATCH_SIZE
        self.device=cfg.DEVICE
                
        self.d_optimizer = optimizer['disc']
        self.g_optimizer = optimizer['gen']

    def dynamicEpochConstruction(self):
        train=self.data_class.returnNewEpoch('train')
        test=self.data_class.returnNewEpoch('test')
        val=self.data_class.returnNewEpoch('val')

        train_fake_mat = GANLoader(self.data_class.data_train,train['fake/mat'],'train')
        train_real_mat = GANLoader(self.data_class.data_train,train['real/mat'],'train')
        train_real_mis=GANLoader(self.data_class.data_train,train['real/mis'],'train')
        train_fake_GP=GANLoader(self.data_class.data_train,train['fake/mat_GP'],'train')

        test_loader = GANLoader(self.data_class.data_test,test,'test')
        val_loader=GANLoader(self.data_class.data_val,val,'val')

        self.dataloader = {
            'train_fake_mat': DataLoader(
            train_fake_mat, 
            batch_size=cfg.GAN_BATCH_SIZE,              
            drop_last=check_dataset(train_fake_mat, cfg.GAN_BATCH_SIZE),
            shuffle=True,
            num_workers=1
            ),
            'train_real_mat': DataLoader(
            train_real_mat, 
            batch_size=cfg.GAN_BATCH_SIZE,              
            drop_last=check_dataset(train_real_mat, cfg.GAN_BATCH_SIZE),
            shuffle=True,
            num_workers=1
            ),
            'train_real_mis': DataLoader(
            train_real_mis, 
            batch_size=cfg.GAN_BATCH_SIZE,              
            drop_last=check_dataset(train_real_mis, cfg.GAN_BATCH_SIZE),
            shuffle=True,
            num_workers=1
            ),
            'train_fake_GP': DataLoader(
            train_fake_GP, 
            batch_size=cfg.GAN_BATCH_SIZE,              
            drop_last=check_dataset(train_real_mis, cfg.GAN_BATCH_SIZE),
            shuffle=True,
            num_workers=1
            ),
            'test': DataLoader(
            test_loader, 
            batch_size=cfg.GAN_BATCH_SIZE,         
            shuffle=True,
            drop_last=check_dataset(test_loader, cfg.GAN_BATCH_SIZE),
            num_workers=1
            ),
            'val': DataLoader(
            val_loader, 
            shuffle=True,
            batch_size=cfg.GAN_BATCH_SIZE,              
            drop_last=check_dataset(val_loader, cfg.GAN_BATCH_SIZE),
            num_workers=1
            )
            }    
            
    def get_infinite_batches(self, dataType):
        while True:
            for i,(_, learned_embedding, voxel) in enumerate(self.dataloader[dataType]):
                yield (learned_embedding.to(cfg.DEVICE),voxel.to(cfg.DEVICE))

                


    def train(self, genSteps):
        scheduler_d = StepLR(self.d_optimizer, step_size=cfg.GAN_DISC_SCHEDULER_STEP, gamma=cfg.GAN_SCHEDULER_GAMMA)
        scheduler_g = StepLR(self.g_optimizer, step_size=cfg.GAN_GEN_SCHEDULER_STEP, gamma=cfg.GAN_SCHEDULER_GAMMA)

        self.dynamicEpochConstruction()
        print("Loading Data...")
        fake_match = self.get_infinite_batches('train_fake_mat')
        real_match=self.get_infinite_batches('train_real_mat')
        real_mismatch=self.get_infinite_batches('train_real_mis')
        fake_GP=self.get_infinite_batches('train_fake_GP')
        print("Training...")
        pbar = tqdm_nb()
        pbar.reset(total=genSteps)
        self.train_log = {
                'generator_loss': [],
                'critic_loss': [],
                #'critic_loss_fake/mat': [],
                #'critic_loss_real/mat': [],
                #'critic_loss_real/mis':[],
                #'critic_loss_gp':[]
                }
        self.val_log ={
                'generator_loss': [],
                'critic_loss': [],
                }
        for genStep in range(genSteps):
            pbar.update()
            desc = 'Generator steps: [%d/%d], Critic steps[%d/%d], g_loss: %f, d_loss %f' \
                    % (genStep+1,genSteps,1,cfg.GAN_NUM_CRITIC_STEPS,0,0)
            pbar.set_description(desc)
            
            self.generator.train()
            self.discriminator.train()
            self.train_log = {
                'generator_loss': [],
                'critic_loss': [],
                #'critic_loss_fake/mat': [],
                #'critic_loss_real/mat': [],
                #'critic_loss_real/mis':[],
                #'critic_loss_gp':[]
                }
            self.val_log ={
                'generator_loss': [],
                'critic_loss': [],
                }
            
            for p in self.discriminator.parameters():
                p.requires_grad = True


            for d_iter in range(cfg.GAN_NUM_CRITIC_STEPS):
                #start=time.time()
                #fake_match
                fake_input_match=fake_match.__next__()
                fake_model=self.generator(fake_input_match[0]) #build fake
                d_out_fake_match = self.discriminator(fake_model['sigmoid_output'],fake_input_match[0])
                #end=time.time()
                #print("fake time ")
                #print(end-start)

                #start=time.time()

                #real_match
                real_input_match=real_match.__next__()
                d_out_real_match = self.discriminator(real_input_match[1],real_input_match[0])
                #end=time.time()
                #print("real time ")
                #print(end-start)


                #start=time.time()
                #real_mismatch
                real_input_mismatch=real_mismatch.__next__()
                d_out_real_mismatch = self.discriminator(real_input_mismatch[1],real_input_mismatch[0])
                #end=time.time()
                #print("real mis time")
                #print(end-start)

                
                if cfg.GAN_GP:
                    fake_input_GP=fake_GP.__next__()
                    fake_input_GP[0].requires_grad=True
                    fake_model_GP=self.generator(fake_input_GP[0])

                    gp_loss=self.calculateGP(fake_input_GP[1],fake_model_GP['sigmoid_output'],fake_input_GP[0])
                else:
                    gp_loss=0
                #start=time.time()

                losses=self.calculateLossDisc(d_out_fake_match,d_out_real_match,d_out_real_mismatch,gp_loss)
                self.train_log['critic_loss'].append(losses['d_loss'].item())
                #self.train_log['critic_loss_fake/mat'].append(losses['d_loss_fake/mat'].item())
                #self.train_log['critic_loss_real/mat'].append(losses['d_loss_real/mat'].item())
                #self.train_log['critic_loss_real/mis'].append(losses['d_loss_real/mis'].item())
                #self.train_log['critic_loss_gp'].append(losses['d_loss_gp'].item())

                self.discriminator.zero_grad()

                losses['d_loss'].backward()

                if not cfg.GAN_GP:
                    clip_grad_value_(self.discriminator.parameters(), cfg.GAN_GRADIENT_CLIPPING)

                self.d_optimizer.step()

                desc = 'Generator steps: [%d/%d], Critic steps[%d/%d], g_loss: %f, d_loss %f' \
                    % (genStep+1,genSteps,d_iter+1,cfg.GAN_NUM_CRITIC_STEPS,0,losses['d_loss'].item())
                pbar.set_description(desc)
                scheduler_d.step()
                #end=time.time()
                #print("loss time")
                #print(end-start)

            for p in self.discriminator.parameters():
                p.requires_grad = False 

            fake_input_match=fake_match.__next__()
            fake_model=self.generator(fake_input_match[0])
            d_loss_fake_match = self.discriminator(fake_model['sigmoid_output'],fake_input_match[0])
            g_loss = torch.mean(-d_loss_fake_match['logits'])  
            self.generator.zero_grad()
   
            g_loss.backward()
            self.g_optimizer.step()
            self.train_log['generator_loss'].append(g_loss.item())
            desc = 'Generator steps: [%d/%d], Critic steps[%d/%d], g_loss: %f, d_loss %f' \
                    % (genStep+1,genSteps,d_iter+1,cfg.GAN_NUM_CRITIC_STEPS,g_loss.item(),losses['d_loss'].item())
            pbar.set_description(desc)

            scheduler_g.step()

            if genStep % cfg.GAN_VAL_PERIOD==0:
                print("Validating...\n")
                self.validate()
                self.val_report(genStep,genSteps)
                

    def calculateGP(self,real_shape,fake_shape,text):
        epsilon=torch.rand((cfg.GAN_BATCH_SIZE,1,1,1,1)).repeat(1,real_shape.shape[1],real_shape.shape[2],real_shape.shape[3],real_shape.shape[4]).to(cfg.DEVICE)
        intepolatedShape=real_shape*epsilon+fake_shape*(1-epsilon)
        mixed_score=self.discriminator(intepolatedShape,text)

        gradient_s=torch.autograd.grad(
            inputs=intepolatedShape,
            outputs=mixed_score['logits'],
            grad_outputs=torch.ones_like(mixed_score['logits']),
            create_graph=True,retain_graph=True
        )[0]
        gradient_t=torch.autograd.grad(
            inputs=text,
            outputs=mixed_score['logits'],
            grad_outputs=torch.ones_like(mixed_score['logits']),
            create_graph=True,retain_graph=True
        )[0]
        gradient_dshape= gradient_s.view(gradient_s.shape[0],-1)

        gradient_s_norm=gradient_dshape.norm(2,dim=1)

        gradient_tshape= gradient_t.view(gradient_t.shape[0],-1)
        
        gradient_t_norm=gradient_tshape.norm(2,dim=1)

        gradint_t_penalty=torch.mean((gradient_t_norm-1)**2)
        gradint_s_penalty=torch.mean((gradient_s_norm-1)**2)

        gp_loss=gradint_s_penalty+gradint_t_penalty
        
        return gp_loss

    def calculateLossDisc(self,fake_critic,mat_critic,mis_critic,gp_loss):

        d_loss_fake_match =fake_critic['logits'].mean() * float(cfg.GAN_FAKE_MATCH_LOSS_COEFF)

        d_loss_real_match =torch.mean(-mat_critic['logits'])* float(cfg.GAN_MATCH_LOSS_COEFF)
        d_loss_real_mismatch = mis_critic['logits'].mean() * float(cfg.GAN_FAKE_MISMATCH_LOSS_COEFF)
        
        

        if not cfg.GAN_GP:
            gp_loss = torch.tensor(0, dtype=torch.float32)
   

        d_loss=d_loss_fake_match+d_loss_real_match+d_loss_real_mismatch+gp_loss

        return {'d_loss':d_loss, 
                'd_loss_fake/mat' : d_loss_fake_match,
                'd_loss_real/mat': d_loss_real_match,
                'd_loss_real/mis':d_loss_real_mismatch,
                'd_loss_gp': gp_loss
        }
    



    def validate(self):
        print("Validating...\n")
        
        pbar = tqdm_nb()
        
        pbar.reset(total=len(self.dataloader['val']))
        self.discriminator.eval()
        self.generator.eval()
        for i,(_,texts,_) in enumerate(self.dataloader['val']):
            pbar.update()
            texts=texts.to(cfg.DEVICE)
            fake_model=self.generator(texts) 
            d_out_fake_match = self.discriminator(fake_model['sigmoid_output'],texts)
            d_loss =fake_model['logits'].mean() * float(cfg.GAN_FAKE_MATCH_LOSS_COEFF)
            g_loss = torch.mean(-d_out_fake_match['logits'])  


            self.val_log['critic_loss'].append(g_loss.item())
            self.val_log['generator_loss'].append(d_loss.item())
    




    def val_report(self,step_id, steps):
        # show report

        print("ganSteps [{}/{}] done...".format(step_id+1, steps))
        print("------------------------summary------------------------")
        print("[train] total_d_loss: %f" % (
                np.mean(self.train_log['critic_loss'])
        ))
        print("[train] total_g_loss: %f" % (
            np.mean(self.train_log['generator_loss'])
        ))

        print("[val]   total_d_loss: %f" % (
            np.mean(self.val_log['critic_loss'])
        ))
        print("[val]   total_g_loss: %f" % (
            np.mean(self.val_log['generator_loss'])
        ))

        