
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
import torch
import numpy as np
from config import cfg
import os
from tqdm.notebook import tqdm as tqdm_nb
from torch.utils.data import DataLoader
from DataLoader.GANloader import GANLoader,check_dataset
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd



class Solver():
    def __init__(self,data, generator,discriminator, optimizer, criterion, batch_size,device):
        self.data=data
        self.generator=generator
        self.discriminator=discriminator
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.criterion = criterion
        self.device=device
        self.BuildDataloaders()
                
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=cfg.GAN_LR, weight_decay=cfg.GAN_WEIGHT_DECAY)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=cfg.GAN_LR, weight_decay=cfg.GAN_WEIGHT_DECAY)

    def BuildDataloaders(self):
        np.random.seed()

        train_fake_match = GANLoader(self.data.train_data,'train',True)

        train_real_match = GANLoader(self.data.train_data,'train',False)

        train_real_mismatch=GANLoader(self.data.train_data,'train',False)

        val_dataset = GANLoader(self.data.val_data,'val',False)

        test_dataset=GANLoader(self.data.test_data,'test',False)

        self.dataloader = {
            'train_fake_match': DataLoader(
            train_fake_match, 
            batch_size=cfg.GAN_BATCH_SIZE,              
            drop_last=check_dataset(train_fake_match, cfg.GAN_BATCH_SIZE),
            shuffle=True,
            num_workers=4
            ),
            'train_real_match': DataLoader(
            train_real_match, 
            batch_size=cfg.GAN_BATCH_SIZE,         
            shuffle=True,
            drop_last=check_dataset(train_real_match, cfg.GAN_BATCH_SIZE),
            num_workers=4
            ),
            'train_real_mismatch': DataLoader(
            train_real_mismatch, 
            shuffle=True,
            batch_size=cfg.GAN_BATCH_SIZE,              
            drop_last=check_dataset(train_real_mismatch, cfg.GAN_BATCH_SIZE),
            num_workers=4
            ),
            'val': DataLoader(
            val_dataset, 
            batch_size=cfg.GAN_BATCH_SIZE,
            num_workers=4
            ),
            'test': DataLoader(
            test_dataset, 
            batch_size=cfg.GAN_BATCH_SIZE,
            num_workers=2
            )
            }    
        
    def get_infinite_batches(self, dataType):
        while True:
            for i,(_, _, learned_embedding, raw_caption , voxel) in enumerate(self.dataloader[dataType]):
                yield (learned_embedding,raw_caption,voxel)

    def train(self, ganSteps):
        scheduler = StepLR(self.optimizer, step_size=cfg.GAN_SCHEDULER_STEP, gamma=cfg.GAN_SCHEDULER_GAMMA)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        one = one.to(self.device)
        mone = mone.to(self.device)

        for ganStep in range(ganSteps):

            print("Training...")
            pbar = tqdm_nb()
            pbar.reset(total=len(self.dataloader['train_fake_match']))

            self.generator.train()

            train_log = {
                'total_loss': [],
                'generator_loss': [],
                'critic_loss': [],
                'gradint_penalty':[]
                }
            val_log ={
                'total_loss': [],
                'generator_loss': [],
                'critic_loss': [],
                'gradint_penalty':[]
                }

            fake_match = self.get_infinite_batches('train_fake_match')
            real_match=self.get_infinite_batches('train_real_match')
            real_mismatch=self.get_infinite_batches('train_real_mismatch')

            Wasserstein_D=0

            for p in self.D.parameters():
                p.requires_grad = True


            for d_iter in range(cfg.GAN_NUM_CRITIC_STEPS):
                self.discriminator.zero_grad()

                pbar.update()

                #real_match
                real_input_match=real_match.__next__()
                d_loss_real_match = self.discriminator(real_input_match[2],real_input_match[0])
                d_loss_real_match = d_loss_real_match.mean()
                d_loss_real_match.backward(mone)

                #real_mismatch
                real_input_mismatch=real_mismatch.__next__()
                d_loss_real_mismatch = self.discriminator(real_input_mismatch[2],real_input_mismatch[0])
                d_loss_real_mismatch = d_loss_real_mismatch.mean()
                d_loss_real_mismatch.backward(mone)

                #fake_match
                fake_input_match=fake_match.__next__()
                fake_model=self.generator(fake_input_match[0])
                d_loss_fake_match = self.discriminator(fake_model,fake_input_match[0])
                d_loss_fake_match = d_loss_fake_match.mean()
                d_loss_fake_match.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(real_input_match[2], fake_input_match[2])
                gradient_penalty.backward()

                d_loss = d_loss_fake_match - d_loss_real_match + gradient_penalty
                Wasserstein_D=d_loss_real_match-d_loss_fake_match
                self.d_optimizer.step()

                desc = 'Generator steps: [%d/%d], Critic steps[%d/%d]' \
                    % (ganStep+1,ganSteps,d_iter,cfg.GAN_NUM_CRITIC_STEPS)
                pbar.set_description(desc)

            for p in self.D.parameters():
                p.requires_grad = False 

            self.generator.zero_grad()
            fake_input_match=fake_match.__next__()
            fake_model=self.generator(fake_input_match[0])
            d_loss_fake_match = self.discriminator(fake_model,fake_input_match[0])
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()

            desc = 'Generator steps: [%d/%d], Critic steps[%d/%d]' \
                    % (ganStep+1,ganSteps,d_iter,cfg.GAN_NUM_CRITIC_STEPS)
            pbar.set_description(desc)

            scheduler.step()

    def calculate_gradient_penalty(self, real_data, fake_data,embeddings):
        eta = torch.FloatTensor(self.batch_size,1,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size,real_data[1], real_data[2], real_data[3],real_data[4])
        eta=eta.to(self.device)

        interpolated = eta * real_data + ((1 - eta) * fake_data)

        interpolated = interpolated.to(self.device)


        interpolated = Variable(interpolated, requires_grad=True)

        prob_interpolated = self.discriminator(interpolated,embeddings)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * cfg.GAN_LAMBDA_TERM
        return grad_penalty






    def validate(self, val_log):
        print("Validating...\n")
        if cfg.EMBEDDING_SHAPE_ENCODER:
            self.shape_encoder.eval()
        self.text_encoder.eval()

        pbar = tqdm_nb()
        pbar.reset(total=len(self.dataloader['val']))

        for i,(_,labels,texts , _, shapes) in enumerate(self.dataloader['val']):
            pbar.update()

            with torch.no_grad():
                losses = self.forward(shapes, texts, labels)

            #print(losses['total_loss'].item())
            # record
            if cfg.EMBEDDING_SHAPE_ENCODER:

                val_log['total_loss'].append(losses['total_loss'].item())
                val_log['walker_loss_tst'].append(losses['walker_loss_tst'].item())
                val_log['walker_loss_sts'].append(losses['walker_loss_sts'].item())
                val_log['visit_loss_ts'].append(losses['visit_loss_ts'].item())
                val_log['visit_loss_st'].append(losses['visit_loss_st'].item())
                val_log['metric_loss_st'].append(losses['metric_loss_st'].item())
                val_log['metric_loss_tt'].append(losses['metric_loss_tt'].item())
                val_log['shape_norm_penalty'].append(losses['shape_norm_penalty'].item())
            
                val_log['text_norm_penalty'].append(losses['text_norm_penalty'].item())
            else:
                val_log['total_loss'].append(losses['total_loss'].item())
                val_log['metric_loss_tt'].append(losses['metric_loss_tt'].item())
                #val_log['text_norm_penalty'].append(losses['text_norm_penalty'].item())

            desc = 'Validating: [%d/%d], Total loss: %.4f' \
                    % (i+1, len(self.dataloader['val']), losses['total_loss'].item())
            pbar.set_description(desc)


        return val_log
    




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