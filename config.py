from easydict import EasyDict as edict


__C = edict()


cfg = __C


__C.DEVICE='cpu'
__C.SEED=4



#################

###EBEDDING SECTION###
__C.EMBEDDING_TRAIN_SPLIT='./../shapenet/processed_captions_train.p'
__C.EMBEDDING_TEST_SPLIT='./../shapenet/processed_captions_test.p'
__C.EMBEDDING_VAL_SPLIT='./../shapenet/processed_captions_val.p'
__C.EMBEDDING_BAD_IDS='./../shapenet/problematic_nrrds_shapenet_unverified_256_filtered_div_with_err_textures.p'
__C.EMBEDDING_SHAPENET='./../shapenet/shapenet.json'

__C.EMBEDDING_CAPTION_CSV="captions.tablechair.csv"

__C.EMBEDDING_TEXT_MODELS_PATH='./../SavedModels'
__C.EMBEDDING_SAVE_PATH='./../GeneratedEmbeddings'
__C.EMBEDDING_SAVE_PATH_TEST='./../GeneratedEmbeddingsTest'
__C.EMBEDDING_INFO_DATA='./../InfoData'
__C.EMBEDDING_SHAPE_ENCODER=False
#GENERAL SETTING
__C.EMBEDDING_BATCH_SIZE=256
__C.EMBEDDING_DIM=128
__C.EMBEDDING_EPOCH_NR=160 
__C.EMBEDDING_CAPTION_LEN=19 
__C.EMBEDDING_ALBERT=False

__C.EMBEDDING_LR=0.001
__C.EMBEDDING_WEIGHT_DC=0.0001

__C.EMBEDDING_SCHEDULER_STEP=20
__C.EMBEDDING_SCHEDULER_GAMMA=0.1

__C.EMBEDDING_GRADIENT_CLIPPING=10.

#LOSS PARAMETERS
__C.METRIC_WEIGHT=0.2
__C.METRIC_MARGIN=0.05
__C.METRIC_MARGIN_TRIPLET=0.7
__C.EPS=1e-10
__C.MAX_NORM=10. #30
__C.TEXT_NORM_MULTIPLIER =1.0
__C.TRIPLET_MULTIPLIER=3.0

###################


###GAN SECTION###
__C.GAN_TRAIN_SPLIT='./../GeneratedEmbeddings/train.p'
__C.GAN_TEST_SPLIT='./../GeneratedEmbeddingsTest/test.p'
__C.GAN_VAL_SPLIT='./../GeneratedEmbeddings/val.p'
__C.GAN_VOXEL_FOLDER='./../nrrd_256_filter_div_32_solid'
__C.GAN_MODELS_PATH='./../SavedModels'
__C.GAN_INFO_DATA='./../InfoData'



__C.GAN_VOXEL_CLIP=0.6

__C.GAN_BATCH_SIZE=32
__C.GAN_GEN_SCHEDULER_STEP=10
__C.GAN_DISC_SCHEDULER_STEP=10

__C.GAN_SCHEDULER_GAMMA=0.1
__C.GAN_LR=0.0001
__C.GAN_WEIGHT_DECAY=0.0001

__C.GAN_NUM_CRITIC_STEPS=5
__C.GAN_NOISE_SIZE = 32
__C.GAN_NOISE_DIST = 'uniform'
__C.GAN_NOISE_MEAN=0.
__C.GAN_NOISE_STDDEV = 0.5
__C.GAN_NOISE_UNIF_ABS_MAX = 1.
__C.GAN_TRAIN_AUGMENT_MAX=10.

__C.GAN_GP=True
__C.GAN_LAMBDA_TERM = 10.
__C.GAN_VAL_PERIOD=150.

__C.GAN_GRADIENT_CLIPPING=0.1

__C.GAN_MATCH_LOSS_COEFF = 1.
__C.GAN_FAKE_MATCH_LOSS_COEFF = 1.
__C.GAN_FAKE_MISMATCH_LOSS_COEFF = 1.

