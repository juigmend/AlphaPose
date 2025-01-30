from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'detector/yolo/cfg/yolov3-spp.cfg'
cfg.WEIGHTS = 'detector/yolo/data/yolov3-spp.weights'
cfg.INP_DIM = 608 # multiple of 32, original = 608, (fast = 420, makes the program stall)
cfg.NMS_THRES = 0.6 # original = 0.6, fast = 0.45
cfg.CONFIDENCE = 0.1 # original = 0.1, fast = 0.5
cfg.NUM_CLASSES = 80 # original = 80
# 'original' refers to the values when forking, the system works well with those values
# 'fast' refers to suggested values for fast performance in the documentation when forking
