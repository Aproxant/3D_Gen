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
        #raw_embedding_list=[]
        #learned_embedding_list = []
        #label_list = []
        #model_list = []
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
                #curr_label_fake=caption_data_fake[1]
                #cur_raw_embedding_fake=caption_data_fake[2]
                #curr_caption_fake=caption_data_fake[3]

                if cur_model_id_mismatch == curr_model_id:  
                    continue  
                inds2.append(db_ind_mismatch)
                break


            #model_list.append(curr_model_id)
            #label_list.append(curr_label_fake)
            #raw_embedding_list.append(curr_caption_fake)
            #learned_embedding_list.append(cur_raw_embedding_fake)

        
        return (inds1,inds2)#(model_list,label_list,learned_embedding_list,raw_embedding_list)

    
    def get_match_batch(self,db_inds,phase):
        #learned_embedding_list = []
        #raw_embedding_list=[]
        #label_list = []
        #model_list = []
        inds=[]
        for ind in db_inds:
            #elem_data = getattr(self, "data_{}".format(phase))[ind]
            #model_list.append(elem_data[0])
            #label_list.append(elem_data[1])
            #learned_embedding_list.append(elem_data[2])
            #raw_embedding_list.append(elem_data[3])
            inds.append(ind)
        return (inds,inds)#(model_list,label_list,learned_embedding_list,raw_embedding_list)
    

    
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
            #(model_list_fake_match, label_batch_fake_match,learned_embedding_batch_fake_match,
            # raw_embedding_batch_fake_match) = self.get_match_batch(db_inds[0],phase)
            (fake_match1,fake_match2) = self.get_match_batch(db_inds[0],phase)
            
            # real / matching
            #(model_list_real_match, label_batch_real_match,learned_embedding_batch_real_match,
            # raw_embedding_batch_real_match) = self.get_match_batch(db_inds[1],phase)
            (real_match1,real_match2) = self.get_match_batch(db_inds[1],phase)

            # real / mismatching
            #(model_list_real_mismatch, label_batch_real_mismatch,learned_embedding_batch_real_mismatch,
            # raw_embedding_batch_real_mismatch)  = self.get_real_mismatch_batch(db_inds[2],num_data,phase)
            (real_mis1,real_mis2)=self.get_real_mismatch_batch(db_inds[2],num_data,phase)

            #fake/match gp
            (fake_gp1,fake_gp2) = self.get_match_batch(db_inds[3],phase)


            self.newGenBatch['fake/mat'].extend(list(zip(fake_match1,fake_match2)))
            #self.newGenBatch['fake/mat'].extend(list(zip(model_list_fake_match, label_batch_fake_match,learned_embedding_batch_fake_match,
            # raw_embedding_batch_fake_match)))
            self.newGenBatch['real/mat'].extend(list(zip(real_match1,real_match2)))
            #self.newGenBatch['real/mat'].extend(list(zip(model_list_real_match, label_batch_real_match,learned_embedding_batch_real_match,
            # raw_embedding_batch_real_match)))
            self.newGenBatch['real/mis'].extend(list(zip(real_mis1,real_mis2)))
            #self.newGenBatch['real/mis'].extend(list(zip(model_list_real_mismatch, label_batch_real_mismatch,learned_embedding_batch_real_mismatch,
            # raw_embedding_batch_real_mismatch)))
            self.newGenBatch['fake/mat_GP'].extend(list(zip(fake_gp1,fake_gp2)))



    def returnNewEpoch(self,phase):
        self.buildBatch(phase)
        return self.newGenBatch

        