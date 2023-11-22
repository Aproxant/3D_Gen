from easydict import EasyDict as edict


__C = edict()


cfg = __C


__C.DEVICE='cpu'



#################

###GAN SECTION###
__C.GAN_TRAIN_SPLIT='./../GeneratedEmbeddings/train.p'
__C.GAN_TEST_SPLIT='./../GeneratedEmbeddings/test.p'
__C.GAN_VAL_SPLIT='./../GeneratedEmbeddings/val.p'
__C.GAN_VOXEL_FOLDER='./../nrrd_256_filter_div_32_solid'
__C.GAN_MODELS_PATH='./../SavedModels'
__C.GAN_INFO_DATA='./../InfoData'

__C.SEED=4

__C.GAN_BATCH_SIZE=32
__C.GAN_GEN_SCHEDULER_STEP=10
__C.GAN_DISC_SCHEDULER_STEP=50

__C.GAN_SCHEDULER_GAMMA=0.1
__C.GAN_LR=0.0001
__C.GAN_WEIGHT_DECAY=0.0001

__C.GAN_NUM_CRITIC_STEPS=5
__C.GAN_NOISE_SIZE = 32
__C.GAN_NOISE_DIST = 'uniform'
__C.GAN_NOISE_MEAN=0.
__C.GAN_NOISE_STDDEV = 1.
__C.GAN_NOISE_UNIF_ABS_MAX = 1.
__C.GAN_TRAIN_AUGMENT_MAX=10.

__C.GAN_GP=True
__C.GAN_LAMBDA_TERM = 10.
__C.GAN_VAL_PERIOD=10.

__C.GAN_GRADIENT_CLIPPING=0.1

__C.GAN_MATCH_LOSS_COEFF = 2.
__C.GAN_FAKE_MATCH_LOSS_COEFF = 1.
__C.GAN_FAKE_MISMATCH_LOSS_COEFF = 1.

"""
        # Train critic more than generator
        for _ in range(num_critic_steps):
            minibatch = self.get_minibatch(train_queue, data_timer=data_timer)
            feed_dict, _ = self.get_feed_dict(minibatch)
            d_loss, d_fake, d_match, d_mismatch = self.discriminator_step(sess, feed_dict, step)

        # Update the generator once every time step
        minibatch = self.get_minibatch(train_queue, data_timer=data_timer)
        feed_dict, _ = self.get_feed_dict(minibatch)
        g_loss = self.generator_step(sess, feed_dict, step)
"""


