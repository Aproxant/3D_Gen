
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_value_
import torch
import numpy as np
from config import cfg
import os
from tqdm.notebook import tqdm as tqdm_nb
from torch.utils.data import DataLoader
from DataLoader.GANloader import GANLoader,check_dataset
from tqdm import tqdm
import pickle
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class SolverGAN():
    def __init__(self,data_class, generator,discriminator, optimizer):
        self.data_class=data_class
        self.generator=generator
        self.discriminator=discriminator
        self.batch_size = cfg.GAN_BATCH_SIZE
        self.device=cfg.DEVICE
                
        self.d_optimizer = optimizer['disc']
        self.g_optimizer = optimizer['gen']

        self.EpochLoss=np.full(5,np.inf)

        self.saveLosses={'train_gen_loss':[],
                         'train_disc_loss':[],
                         'val_gen_loss':[],
                         'val_disc_loss':[],
                         'step':[]}

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
            num_workers=4,pin_memory=True
            ),
            'train_fake_GP': DataLoader(
            train_fake_GP, 
            batch_size=cfg.GAN_BATCH_SIZE,              
            drop_last=check_dataset(train_real_mis, cfg.GAN_BATCH_SIZE),
            shuffle=True,
            num_workers=4,pin_memory=True
            ),
            'test': DataLoader(
            test_loader, 
            batch_size=cfg.GAN_BATCH_SIZE,         
            shuffle=True,
            drop_last=check_dataset(test_loader, cfg.GAN_BATCH_SIZE),
            num_workers=4,pin_memory=True
            ),
            'val': DataLoader(
            val_loader, 
            shuffle=True,
            batch_size=cfg.GAN_BATCH_SIZE,              
            drop_last=check_dataset(val_loader, cfg.GAN_BATCH_SIZE),
            num_workers=4,pin_memory=True
            )
            }    
            
    def get_infinite_batches(self, dataType):
        while True:
            for i,(_, learned_embedding, voxel) in enumerate(self.dataloader[dataType]):
                yield (learned_embedding.to(cfg.DEVICE),voxel.to(cfg.DEVICE))


    def train(self, epochs):
        scheduler_d = StepLR(self.d_optimizer, step_size=cfg.GAN_DISC_SCHEDULER_STEP, gamma=cfg.GAN_SCHEDULER_GAMMA)
        scheduler_g = StepLR(self.g_optimizer, step_size=cfg.GAN_GEN_SCHEDULER_STEP, gamma=cfg.GAN_SCHEDULER_GAMMA)

        self.dynamicEpochConstruction()
        print("Loading Data...")
        fake_match = self.get_infinite_batches('train_fake_mat')
        #real_match=self.get_infinite_batches('train_real_mat')
        #real_mismatch=self.get_infinite_batches('train_real_mis')
        fake_GAN=self.get_infinite_batches('train_fake_GP')
        
        print("Training...")
        genSteps=len( self.dataloader['train_fake_mat'])
        for epoch_id in range(epochs):
            print("Epoch [{}/{}] starting...\n".format(epoch_id+1, epochs))
            #pbar = tqdm_nb()
            #pbar.reset(total=genSteps)

            for genStep in tqdm(range(genSteps)):
                self.train_log = {
                'generator_loss': [],
                'critic_loss': [],
                }
                self.val_log ={
                'generator_loss': [],
                'critic_loss': [],
                }
                #pbar.update()
                desc = 'Generator steps: [%d/%d], Critic steps[%d/%d], g_loss: %f, d_loss %f' \
                    % (genStep+1,genSteps,1,cfg.GAN_NUM_CRITIC_STEPS,0,0)
                #pbar.set_description(desc)
            
                self.generator.train()
                self.discriminator.train()

            


                crit_loss=[]
                for d_iter in range(cfg.GAN_NUM_CRITIC_STEPS):
                    
                    #fake_match
                    fake_input_match=fake_match.__next__()
                    
                    fake_model=self.generator(fake_input_match[0]) #build fake

                    d_out_fake_match = self.discriminator(fake_model['sigmoid_output'],fake_input_match[0])

                    #real_match
                    #real_input_match=real_match.__next__()
                    d_out_real_match = self.discriminator(fake_input_match[1],fake_input_match[0])

                    ##real_mismatch
                    #real_input_mismatch=real_mismatch.__next__()
                    #d_out_real_mismatch = self.discriminator(real_input_mismatch[1],real_input_mismatch[0])
                
                    if cfg.GAN_GP:
                        #fake_input_GP=fake_GP.__next__()
                        #fake_input_GP[0].requires_grad=True
                        #fake_model_GP=self.generator(fake_input_GP[0])
                        gp_loss=self.calculateGP(fake_input_match[1],fake_model['sigmoid_output'],fake_input_match[0])
                    else:
                        gp_loss=0

                    losses=self.calculateLossDisc(d_out_fake_match,d_out_real_match,None,gp_loss)
                    crit_loss.append(losses['d_loss'].item())

                    self.discriminator.zero_grad()

                    losses['d_loss'].backward()

                    if not cfg.GAN_GP:
                        clip_grad_value_(self.discriminator.parameters(), cfg.GAN_GRADIENT_CLIPPING)

                    self.d_optimizer.step()

                    desc = 'Generator steps: [%d/%d], Critic steps[%d/%d], g_loss: %f, d_loss %f' \
                    % (genStep+1,genSteps,d_iter+1,cfg.GAN_NUM_CRITIC_STEPS,0,losses['d_loss'].item())
                    #pbar.set_description(desc)
                    self.train_log['critic_loss'].append(losses['d_loss'].item())


                #self.train_log['critic_loss'].append(sum(crit_loss)/cfg.GAN_NUM_CRITIC_STEPS)

                fake_input_match=fake_GAN.__next__()
                fake_model=self.generator(fake_input_match[0])
                d_loss_fake_match = self.discriminator(fake_model['sigmoid_output'],fake_input_match[0])
                g_loss = -1.0*torch.mean(d_loss_fake_match['logits'])  

                self.generator.zero_grad()
   
                g_loss.backward()
                self.g_optimizer.step()

                self.train_log['generator_loss'].append(g_loss.item())
                desc = 'Generator steps: [%d/%d], Critic steps[%d/%d], g_loss: %f, d_loss %f' \
                    % (genStep+1,genSteps,d_iter+1,cfg.GAN_NUM_CRITIC_STEPS,g_loss.item(),losses['d_loss'].item())
                #pbar.set_description(desc)

                scheduler_g.step()
                scheduler_d.step()
                
                print("Generator Loss: "+str(g_loss.item())+ "|  Critic Loss :"+ str(np.mean(self.train_log['critic_loss'])))

                if genStep % cfg.GAN_VAL_PERIOD==0 and genStep!=0:
                    print("Validating...\n")
                    self.validate()
                    self.val_report(genStep,genSteps)
                    with open(os.path.join(cfg.GAN_INFO_DATA,'GAN_data.pkl'), 'wb') as fp:
                        pickle.dump(self.saveLosses, fp)
            #        self.saveLosses['train_gen_loss'].append(np.mean(self.train_log['generator_loss']))
            #        self.saveLosses['train_disc_loss'].append(np.mean(self.train_log['critic_loss']))
            #        self.saveLosses['step'].append(genStep+(epoch_id*genSteps))
                self.saveLosses['train_gen_loss'].append(np.mean(self.train_log['generator_loss']))
                self.saveLosses['train_disc_loss'].append(np.mean(self.train_log['critic_loss']))
                self.saveLosses['step'].append(genStep)
                
            torch.save(self.discriminator.state_dict(), os.path.join(cfg.GAN_MODELS_PATH,"discriminator_model.pth"))
            torch.save(self.generator.state_dict(),os.path.join(cfg.GAN_MODELS_PATH,"generator_model.pth"))
            self.saveModel()

            #pbar.close()
                

    def calculateGP(self,real_shape,fake_shape,text):
        epsilon=torch.rand((cfg.GAN_BATCH_SIZE,1,1,1,1)).repeat(1,real_shape.shape[1],real_shape.shape[2],real_shape.shape[3],real_shape.shape[4]).to(cfg.DEVICE)
        intepolatedShape=real_shape*epsilon+fake_shape*(1-epsilon)
        intepolatedShape=Variable(intepolatedShape,requires_grad=True).to(cfg.DEVICE)
        mixed_score=self.discriminator(intepolatedShape,text)['logits']

        gradients=torch_grad(outputs=mixed_score,inputs=intepolatedShape,
                             grad_outputs=torch.ones(mixed_score.size()).to(cfg.DEVICE),
                             create_graph=True,retain_graph=True)[0]
        
        gradients = gradients.view(cfg.GAN_BATCH_SIZE, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return cfg.GAN_LAMBDA_TERM*((gradients_norm - 1) ** 2).mean()
    """
        gradient_s=torch.autograd.grad(
            inputs=intepolatedShape,
            outputs=mixed_score['logits'],
            grad_outputs=torch.ones_like(mixed_score['logits']),
            create_graph=True,retain_graph=True
        )[0]

        gradient_dshape= gradient_s.view(gradient_s.shape[0],-1)

        gradient_s_norm=gradient_dshape.norm(2,dim=1)

        #gradient_tshape= gradient_t.view(gradient_t.shape[0],-1)
        
        #gradient_t_norm=gradient_tshape.norm(2,dim=1)

        #gradint_t_penalty=torch.mean((gradient_t_norm-1)**2)
        gradint_s_penalty=torch.mean((gradient_s_norm-1)**2)

        gp_loss=gradint_s_penalty#+gradint_t_penalty
        
        return gp_loss
    """
    def calculateLossDisc(self,fake_critic,mat_critic,mis_critic,gp_loss):

        d_loss_fake_match =fake_critic['logits'].mean() * float(cfg.GAN_FAKE_MATCH_LOSS_COEFF)

        d_loss_real_match =mat_critic['logits'].mean()* float(cfg.GAN_MATCH_LOSS_COEFF)
        #d_loss_real_mismatch = mis_critic['logits'].mean() * float(cfg.GAN_FAKE_MISMATCH_LOSS_COEFF)
        
        test_loss=torch.mean(fake_critic['logits'])-torch.mean(mat_critic['logits'])

        if not cfg.GAN_GP:
            gp_loss = torch.tensor(0, dtype=torch.float32)
   

        d_loss=test_loss+gp_loss #tu minus zamieniony
        #d_loss=test_loss+gp_loss
        return {'d_loss':d_loss, 
                'd_loss_fake/mat' : d_loss_fake_match,
                'd_loss_real/mat': d_loss_real_match,
                #'d_loss_real/mis':d_loss_real_mismatch,
                'd_loss_gp': gp_loss
        }
    



    def validate(self):
        print("Validating...\n")
        
        #pbar = tqdm_nb()
        
        #pbar.reset(total=len(self.dataloader['val']))
        self.discriminator.eval()
        self.generator.eval()
        for i,(_,texts,_) in tqdm(enumerate(self.dataloader['val']),total=len(self.dataloader['val'])):
            #pbar.update()
            texts=texts.to(cfg.DEVICE)
            fake_model=self.generator(texts) 
            d_out_fake_match = self.discriminator(fake_model['sigmoid_output'],texts)
            d_loss =fake_model['logits'].mean() * float(cfg.GAN_FAKE_MATCH_LOSS_COEFF)
            g_loss = torch.mean(-d_out_fake_match['logits'])  


            self.val_log['critic_loss'].append(g_loss.item())
            self.val_log['generator_loss'].append(d_loss.item())

        self.saveLosses['val_gen_loss'].append(np.mean(self.val_log['generator_loss']))
        self.saveLosses['val_disc_loss'].append(np.mean(self.val_log['critic_loss']))
        #pbar.close()


    def saveModel(self):
        epoch_loss=0#sum(self.train_log['generator_loss'])/len(self.train_log['generator_loss'])
        if all(self.EpochLoss<epoch_loss):
            print('Stop training')
            return
        else:
            if min(self.EpochLoss)>epoch_loss:
                print("Saving models...\n")

                torch.save(self.discriminator.state_dict(), os.path.join(cfg.GAN_MODELS_PATH,"discriminator_model.pth"))
                torch.save(self.generator.state_dict(),os.path.join(cfg.GAN_MODELS_PATH,"generator_model.pth"))

            newLoss=[]
            for i in range(4):
                newLoss.append(self.EpochLoss[i+1])

            newLoss.append(epoch_loss)
            self.EpochLoss=newLoss



        




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

        