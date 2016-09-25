__author__ = 'stephen'

from easydict import EasyDict

__C = EasyDict()

# python layer file can import layer_config.cfg to get the default configuration
cfg = __C

__C.TRAIN = EasyDict()

__C.TRAIN.INPUT_SIZE = 107

# 256 samples for each image
__C.TRAIN.IMS_PER_BATCH = 64

__C.TRAIN.PARAMS = (0.2, 0.2, 0.05, 0.7, 0.5)
