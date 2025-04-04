from easydict import EasyDict as edict

# JIMG = edited by Juan Ignacio Mendoza Garay

cfg = edict()
cfg.CONFIG = 'detector/yolo/cfg/yolov3-spp.cfg' # '' # JIMG
cfg.WEIGHTS = 'detector/yolo/data/yolov3-spp.weights'
# r'\detector\yolox\data\yolox_x.pth' # JIMG
# r'\detector\yolox\data\yolox_l.pth' # JIMG
#   'detector/yolo/data/yolov3-spp.weights' # JIMG
cfg.INP_DIM = 640 # 2560, 1536, 1280, 1024, 960, 768, 736, 704, 672, 640 (good for 4 people) # JIMG
                   # multiple of 32, original = 608, ("fast" = 420, might stall) # JIMG
cfg.NMS_THRES = 0.6 # original = 0.6, fast = 0.45 # JIMG
cfg.CONFIDENCE = 0.1 # original = 0.1, fast = 0.5 # JIMG
cfg.NUM_CLASSES = 80 # original = 80 # JIMG
# JIMG:
# 'original' refers to the values when forking, the system works well with those values
# 'fast' refers to suggested values for fast performance in the documentation when forking
# BEST: yolov3_640_06_01_80