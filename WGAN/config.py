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


__C.GAN_BATCH_SIZE=256
