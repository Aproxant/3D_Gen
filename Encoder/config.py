from easydict import EasyDict as edict


__C = edict()


cfg = __C


__C.DEVICE='cpu'



#################

###EBEDDING SECTION###
__C.EMBEDDING_TRAIN_SPLIT='./../shapenet/processed_captions_train.p'
__C.EMBEDDING_TEST_SPLIT='./../shapenet/processed_captions_test.p'
__C.EMBEDDING_VAL_SPLIT='./../shapenet/processed_captions_val.p'
__C.EMBEDDING_BAD_IDS='./../shapenet/problematic_nrrds_shapenet_unverified_256_filtered_div_with_err_textures.p'
__C.EMBEDDING_SHAPENET='./../shapenet/shapenet.json'

__C.EMBEDDING_CAPTION_CSV="captions.tablechair.csv"

__C.EMBEDDING_TEXT_MODELS_PATH='./../SavedModels/text_encoder.pth'
__C.EMBEDDING_SAVE_PATH='./../GeneratedEmbeddings'

__C.EMBEDDING_SHAPE_ENCODER=False
#GENERAL SETTING
__C.EMBEDDING_BATCH_SIZE=128 
__C.EMBEDDING_DIM=128
__C.EMBEDDING_EPOCH_NR=20 # ziwekszyc
__C.EMBEDDING_CAPTION_LEN=15
__C.EMBEDDING_ALBERT=False

__C.EMBEDDING_LR=0.001
__C.EMBEDDING_WEIGHT_DC=0.0001

__C.EMBEDDING_SCHEDULER_STEP=5
__C.EMBEDDING_SCHEDULER_GAMMA=0.1

__C.EMBEDDING_GRADIENT_CLIPPING=5.

#LOSS PARAMETERS
__C.METRIC_WEIGHT=1.0
__C.METRIC_MARGIN=0.1
__C.METRIC_MARGIN_TRIPLET=0.6
__C.EPS=1e-10
__C.MAX_NORM=15.
__C.TEXT_NORM_MULTIPLIER =1.0
__C.TRIPLET_MULTIPLIER=1.0
__C.SEED=4

###################