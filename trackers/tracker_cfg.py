from easydict import EasyDict as edict
cfg = edict()
cfg.nid = 1000
cfg.arch = "osnet_ain" # "osnet" or "res50-fc512" # default = "osnet_ain" (JIMG)
cfg.loadmodel = "trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
cfg.frame_rate =  30 # default = 30 (JIMG)
cfg.track_buffer = 240 # default = 240 (JIMG)
cfg.conf_thres = 0.5 # default = 0.5 (JIMG)
cfg.nms_thres = 0.4 # default = 0.4 (JIMG)
cfg.iou_thres = 0.5 # default = 0.5 (JIMG)
