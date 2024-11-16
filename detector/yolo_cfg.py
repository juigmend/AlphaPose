from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = 'detector/yolo/cfg/yolov3-spp.cfg'
cfg.WEIGHTS = 'detector/yolo/data/yolov3-spp.weights'
cfg.INP_DIM = 608 # default = 608, (fast = 420)
cfg.NMS_THRES = 0.6 # default = 0.6, fast = 0.45
cfg.CONFIDENCE = 0.1 # default = 0.1, fast = 0.5
cfg.NUM_CLASSES = 80 # default = 80
