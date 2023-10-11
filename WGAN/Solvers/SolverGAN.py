
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

        train_fake_mat = GANLoader(self.data_class.data_val,train['fake/mat'],'train')
        train_real_mat = GANLoader(self.data_class.data_val,train['real/mat'],'train')
        train_real_mis=GANLoader(self.data_class.data_val,train['real/mis'],'train')
        test_loader = GANLoader(self.data_class.data_val,test,'test')
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
            for i,(_, _, learned_embedding, raw_caption , voxel) in enumerate(self.dataloader[dataType]):
                yield (learned_embedding.to(cfg.DEVICE),raw_caption.to(cfg.DEVICE),voxel.to(cfg.DEVICE))

                


    def train(self, genSteps):
        scheduler_d = StepLR(self.d_optimizer, step_size=cfg.GAN_DISC_SCHEDULER_STEP, gamma=cfg.GAN_SCHEDULER_GAMMA)
        scheduler_g = StepLR(self.g_optimizer, step_size=cfg.GAN_GEN_SCHEDULER_STEP, gamma=cfg.GAN_SCHEDULER_GAMMA)

        self.dynamicEpochConstruction()

        fake_match = self.get_infinite_batches('train_fake_mat')
        real_match=self.get_infinite_batches('train_real_mat')
        real_mismatch=self.get_infinite_batches('train_real_mis')

        print("Training...")
        pbar = tqdm_nb()
        pbar.reset(total=genSteps)
        for genStep in range(genSteps):
            pbar.update()
            desc = 'Generator steps: [%d/%d], Critic steps[%d/%d], g_loss: %f, d_loss %f' \
                    % (genStep+1,genSteps,1,cfg.GAN_NUM_CRITIC_STEPS,0,0)
            pbar.set_description(desc)
            
            self.generator.train()
            self.discriminator.train()
            train_log = {
                'generator_loss': [],
                'critic_loss': [],
                'critic_loss_fake/mat': [],
                'critic_loss_real/mat': [],
                'critic_loss_real/mis':[],
                'critic_loss_gp':[]
                }
            val_log ={
                'generator_loss': [],
                'critic_loss': [],
                'critic_loss_fake/mat': [],
                'critic_loss_real/mat': [],
                'critic_loss_real/mis':[],
                'critic_loss_gp':[]
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
                d_out_real_match = self.discriminator(real_input_match[2],real_input_match[0])
                #end=time.time()
                #print("real time ")
                #print(end-start)


                #start=time.time()
                #real_mismatch
                real_input_mismatch=real_mismatch.__next__()
                d_out_real_mismatch = self.discriminator(real_input_mismatch[2],real_input_mismatch[0])
                #end=time.time()
                #print("real mis time")
                #print(end-start)

                
                if cfg.GAN_GP:
                    selector=float(np.random.randint(2))
                    gp_shape_data = (selector * fake_model['sigmoid_output']
                             + (1. - selector) * real_input_match[2])
                    gp_text_data = (selector* fake_input_match[0]
                            + (1. - selector) * real_input_match[0])

                    d_out_gp=self.discriminator(gp_shape_data,gp_text_data)

                else:
                    d_out_gp = None
                    gp_shape_data = None
                    gp_text_data = None
                #start=time.time()

                losses=self.calculateLossDisc(d_out_fake_match,d_out_real_match,d_out_real_mismatch,d_out_gp,gp_text_data,gp_shape_data)
                train_log['critic_loss'].append(losses['d_loss'].item())
                train_log['critic_loss_fake/mat'].append(losses['d_loss_fake/mat'].item())
                train_log['critic_loss_real/mat'].append(losses['d_loss_real/mat'].item())
                train_log['critic_loss_real/mis'].append(losses['d_loss_real/mis'].item())
                train_log['critic_loss_gp'].append(losses['d_loss_gp'].item())

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
            train_log['generator_loss'].append(g_loss.item())
            desc = 'Generator steps: [%d/%d], Critic steps[%d/%d], g_loss: %f, d_loss %f' \
                    % (genStep+1,genSteps,d_iter+1,cfg.GAN_NUM_CRITIC_STEPS,g_loss.item(),losses['d_loss'].item())
            pbar.set_description(desc)

            scheduler_g.step()

            if genStep % cfg.GAN_VAL_PERIOD==0:
                print("Validating...\n")
                self.validate()
                
                
                print("Training...")




    def calculateLossDisc(self,fake_critic,mat_critic,mis_critic,gp_critic,gp_text,gp_shape):

        d_loss_fake_match =fake_critic['logits'].mean() * float(cfg.GAN_FAKE_MATCH_LOSS_COEFF)

        d_loss_real_match =torch.mean(-mat_critic['logits'])* float(cfg.GAN_MATCH_LOSS_COEFF)
        d_loss_real_mismatch = mis_critic['logits'].mean() * float(cfg.GAN_FAKE_MISMATCH_LOSS_COEFF)
        
        

        if cfg.GAN_GP:
            
            gradient_gp_t = Variable(gp_text, requires_grad=True).to(cfg.DEVICE)
            gradient_gp_s=Variable(gp_shape, requires_grad=True).to(cfg.DEVICE)
            

            gradients_dtext = torch.autograd.grad(outputs=gp_critic['logits'], inputs=gradient_gp_t,create_graph=True, retain_graph=True)
            
            
            gradients_dshape = torch.autograd.grad(outputs=gp_critic['logits'], inputs=gradient_gp_s, create_graph=True, retain_graph=True)
            
            gradients_dshape_reshaped = gradients_dshape.view(self.batch_size, -1)

            slopes_text = torch.sqrt(torch.sum(gradients_dtext**2, dim=1))
            slopes_shape = torch.sqrt(torch.sum(gradients_dshape_reshaped**2, dim=1))

            gp_text = torch.mean((slopes_text - 1.0)**2)
            gp_shape = torch.mean((slopes_shape - 1.0)**2)

            gradient_penalty = gp_text + gp_shape
            d_loss_gp = float(cfg.GAN_LAMBDA_TERM)* gradient_penalty
        else:
            d_loss_gp = torch.tensor(0, dtype=torch.float32)
   
        #print(d_loss_fake_match)
        #print(d_loss_real_match)
        #print(d_loss_real_mismatch)
        d_loss=d_loss_fake_match+d_loss_real_match+d_loss_real_mismatch+d_loss_gp
        #print(d_loss)
        return {'d_loss':d_loss, 
                'd_loss_fake/mat' : d_loss_fake_match,
                'd_loss_real/mat': d_loss_real_match,
                'd_loss_real/mis':d_loss_real_mismatch,
                'd_loss_gp': d_loss_gp
        }
    



    def validate(self):
        print("Validating...\n")

        val_res=[]
        pbar = tqdm_nb()
        pbar.reset(total=len(self.dataloader['val']))
        for i,(_,_,texts , _, _) in enumerate(self.dataloader['val']):
            pbar.update()
            texts=texts.to(cfg.DEVICE)

            fake_model=self.generator(texts) #build fake
            d_out_fake_match = self.discriminator(fake_model['sigmoid_output'],texts)
            g_loss = torch.mean(-d_out_fake_match['logits'])  
            val_res.append(g_loss.item())

        return val_res
    




    def _epoch_report(self,train_log, val_log, epoch_id, epoch):
        # show report
        if cfg.EMBEDDING_SHAPE_ENCODER:

            print("epoch [{}/{}] done...".format(epoch_id+1, epoch))
            print("------------------------summary------------------------")
            print("[train] total_loss: %f" % (
                np.mean(train_log['total_loss'])
            ))
            print("[val]   total_loss: %f" % (
                np.mean(val_log['total_loss'])
            ))

            print("[train] walker_loss_tst: %f, walker_loss_sts: %f" % (
                np.mean(train_log['walker_loss_tst']),
                np.mean(train_log['walker_loss_sts'])
            ))
            print("[val]   walker_loss_tst: %f, walker_loss_sts: %f" % (
                np.mean(val_log['walker_loss_tst']),
                np.mean(val_log['walker_loss_sts'])
            ))
            print("[train] visit_loss_ts: %f, visit_loss_st: %f" % (
                np.mean(train_log['visit_loss_ts']),
                np.mean(train_log['visit_loss_st'])
            ))
            print("[val]   visit_loss_ts: %f, visit_loss_st: %f" % (
                np.mean(val_log['visit_loss_ts']),
                np.mean(val_log['visit_loss_st'])
            ))
            print("[train] metric_loss_st: %f, metric_loss_tt: %f" % (
                np.mean(train_log['metric_loss_st']),
                np.mean(train_log['metric_loss_tt'])
            ))
            print("[val]   metric_loss_st: %f, metric_loss_tt: %f" % (
                np.mean(val_log['metric_loss_st']),
                np.mean(val_log['metric_loss_tt'])
            ))
            print("[train] shape_norm_penalty: %f, text_norm_penalty: %f" % (
                np.mean(train_log['shape_norm_penalty']),
                np.mean(train_log['text_norm_penalty'])
            ))
            print("[val]   shape_norm_penalty: %f, text_norm_penalty: %f\n" % (
                np.mean(val_log['shape_norm_penalty']),
                np.mean(val_log['text_norm_penalty'])
            ))
        else:
            print("epoch [{}/{}] done...".format(epoch_id+1, epoch))
            print("------------------------summary------------------------")
            print("[train] total_loss: %f" % (
                np.mean(train_log['total_loss'])
            ))
            print("[val]   total_loss: %f" % (
                np.mean(val_log['total_loss'])
            ))
            print("[train] metric_loss_tt: %f" % (
                np.mean(train_log['metric_loss_tt'])
            ))
            print("[val]  metric_loss_tt: %f" % (
                np.mean(val_log['metric_loss_tt'])
            ))