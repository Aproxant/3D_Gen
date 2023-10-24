from config import cfg
import numpy as np
import pickle 

class GANDataGenerator():
    def __init__(self):
        for i in ['train','test','val']:
            with open(getattr(cfg,"GAN_{}_SPLIT".format(i.upper())), 'rb') as pickle_file:
                tmp=pickle.load(pickle_file)
            setattr(self, "data_{}".format(i), tmp)


    
    def get_real_mismatch_batch(self, db_inds,num_data,phase):

        inds1=[]
        inds2=[]
        for ind in db_inds:

            caption_data = getattr(self, "data_{}".format(phase))[ind]

            curr_model_id=caption_data[0]

            inds1.append(ind)
            while True:
                db_ind_mismatch = np.random.randint(num_data)
                caption_data_fake=getattr(self, "data_{}".format(phase))[db_ind_mismatch]

                cur_model_id_mismatch=caption_data_fake[0]


                if cur_model_id_mismatch == curr_model_id:  
                    continue  
                inds2.append(db_ind_mismatch)
                break
        
        return (inds1,inds2)

    
    def get_match_batch(self,db_inds,phase):
        return (db_inds,db_inds)

    
    def shuffle_db_inds(self,num_data):

        self.perms = [np.random.permutation(np.arange(num_data)) for _ in range(4)]

        self.cur = 0

    def get_next_minibatch(self,num_data):
        if (self.cur + cfg.GAN_BATCH_SIZE) >= num_data:
            return None

        db_inds = [perm[self.cur:min(self.cur + cfg.GAN_BATCH_SIZE, num_data)] for perm in self.perms]
        self.cur += cfg.GAN_BATCH_SIZE
        return db_inds

    def buildBatch(self,phase):
        num_data=len(getattr(self, "data_{}".format(phase)))

        self.shuffle_db_inds(num_data)

        self.newGenBatch={'fake/mat': [],
                           'real/mat':[],
                           'real/mis':[],
                           'fake/mat_GP':[]}
        
        if phase!='train':
            self.newGenBatch=[]
            while self.cur < num_data:
                db_inds = self.get_next_minibatch(num_data)
                if db_inds is None:
                    return
                (fake_match1,fake_match2) = self.get_match_batch(db_inds[0],phase)

                self.newGenBatch.extend(list(zip(fake_match1,fake_match2)))

        while self.cur < num_data:
            db_inds = self.get_next_minibatch(num_data)
            if db_inds is None:
                return
            
            # fake / matching

            (fake_match1,fake_match2) = self.get_match_batch(db_inds[0],phase)
            
            # real / matching

            (real_match1,real_match2) = self.get_match_batch(db_inds[1],phase)

            # real / mismatching

            (real_mis1,real_mis2)=self.get_real_mismatch_batch(db_inds[2],num_data,phase)

            #fake/match gp
            (fake_gp1,fake_gp2) = self.get_match_batch(db_inds[3],phase)


            self.newGenBatch['fake/mat'].extend(list(zip(fake_match1,fake_match2)))

            self.newGenBatch['real/mat'].extend(list(zip(real_match1,real_match2)))

            self.newGenBatch['real/mis'].extend(list(zip(real_mis1,real_mis2)))

            self.newGenBatch['fake/mat_GP'].extend(list(zip(fake_gp1,fake_gp2)))



    def returnNewEpoch(self,phase):
        self.buildBatch(phase)
        return self.newGenBatch

        