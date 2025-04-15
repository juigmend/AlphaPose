"""Run Alphapose. This module derives from the modified scripts/demo_inference.py (JIMG)"""

import argparse
import os
import platform
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
import natsort
from easydict import EasyDict as edict

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

def run(argdict):
    '''
    Run AlphaPose.
    Args: dictionary, with the following items:

        General options:
            suffix: str. Add suffix to filenames for output JSON and video. Default="".
            param: float. Detector parameters'idim thre conf'.
            cfg: str. Experiment configure file name.
            checkpoint: str. Checkpoint file name.
            sp: bool. Pytorch single process. Default=False. Always True for Windows OS.
            detector: str. Detector name. Default="yolo"
            detfile: str. Detection result file.
            indir: str. Image-directory.
            list: str. Image list.
            image: str. Image name. Default=""
            jsonoutdir: str. Output directory for json files.
            visoutdir: str. Output directory for video and image files.
            save_img: bool. Save result as image. Default=False
            vis: bool. Visualise image. Default=False
            showbox: bool. Visualize human bbox. Default=False
            profile: bool. Add speed profiling at screen output. Default=False
            format: str. Save in the format of cmu or coco or openpose, option: coco/cmu/open'.
            min_box_area: int. Minimum box area to filter out. Default=0
            detbatch: int. Detection batch size per GPU. Default=5
            posebatch: int. Pose estimation maximum batch size per GPU. Default=64
            eval: dest='eval: bool, Save the result json as coco format,
                  using image index(int) instead of image name(str). Default=False
            gpus: str. Index of CUDA device. Comma to use several, e.g. "0,1,2,3".
                       Use "-1" for cpu only. Default="0"
            qsize: int. Length of result buffer (cpu memory). Default=1024
            flip: bool. Enable flip testing. Default=False
            debug: bool. Print detailed information. Default=False
            verbosity: 0, 1, or 2. Default = 2

        Video options:
            video: str. Video file name.
            webcam: int. Webcam number.
            save_video: bool. Save rendered video. Default=False
            vis_fast: bool. Fast rendering (real time). Default=False.

        Tracking options:
            pose_flow: bool. Track humans in video with PoseFlow. Default=False
            pose_track: bool. Track humans in video with reid. Default=False
    '''
    args = edict()
    args.suffix = argdict.get('suffix','')
    args.param = argdict.get('param',None)
    args.cfg = argdict.get('cfg',None)
    args.checkpoint = argdict.get('checkpoint',None)
    args.sp = argdict.get('sp',False)
    args.detector = argdict.get('detector','yolo')
    args.detfile = argdict.get('detfile','')
    args.inputpath = argdict.get('indir','')
    args.inputlist = argdict.get('list','')
    args.inputimg = argdict.get('image','')
    args.outputpath = argdict.get('jsonoutdir','')
    args.visoutpath = argdict.get('visoutdir','')
    args.save_img = argdict.get('save_img',False)
    args.vis = argdict.get('vis',False)
    args.showbox = argdict.get('showbox',False)
    args.profile = argdict.get('profile',False)
    args.format = argdict.get('format',None)
    args.min_box_area = argdict.get('min_box_area',0)
    args.detbatch = argdict.get('detbatch',5)
    args.posebatch = argdict.get('posebatch',64)
    args.eval = argdict.get('eval',False)
    args.gpus = argdict.get('gpus','0')
    args.qsize = argdict.get('qsize',1024)
    args.flip = argdict.get('flip',False)
    args.verbosity = argdict.get('verbosity',2)
    args.debug = argdict.get('debug',False)
    args.video = argdict.get('video','')
    args.webcam = argdict.get('webcam',-1)
    args.save_video = argdict.get('save_video',False)
    args.vis_fast = argdict.get('vis_fast',False)
    args.pose_flow = argdict.get('pose_flow',False)
    args.pose_track = argdict.get('pose_track',False)


    cfg = update_config(args.cfg)

    if platform.system() == 'Windows':
        args.sp = True

    args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
    args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
    args.detbatch = args.detbatch * len(args.gpus)
    args.posebatch = args.posebatch * len(args.gpus)
    args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

    if not args.sp:
        torch.multiprocessing.set_start_method('forkserver', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')

    def check_input():
        # for webcam
        if args.webcam != -1:
            args.detbatch = 1
            return 'webcam', int(args.webcam)

        # for video
        if len(args.video):
            if os.path.isfile(args.video):
                videofile = args.video
                return 'video', videofile
            else:
                raise IOError('Error: --video must refer to a video file, not directory.')

        # for detection results
        if len(args.detfile):
            if os.path.isfile(args.detfile):
                detfile = args.detfile
                return 'detfile', detfile
            else:
                raise IOError('Error: --detfile must refer to a detection json file, not directory.')

        # for images
        if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
            inputpath = args.inputpath
            inputlist = args.inputlist
            inputimg = args.inputimg

            if len(inputlist):
                im_names = open(inputlist, 'r').readlines()
            elif len(inputpath) and inputpath != '/':
                for root, dirs, files in os.walk(inputpath):
                    im_names = files
                im_names = natsort.natsorted(im_names)
            elif len(inputimg):
                args.inputpath = os.path.split(inputimg)[0]
                im_names = [os.path.split(inputimg)[1]]

            return 'image', im_names

        else:
            raise NotImplementedError

    def print_finish_info():
        if args.verbosity==2:
            print('Pose detection and tracking done.')
        if (args.save_img or args.save_video) and not args.vis_fast:
            print('Rendering remaining images in the queue.')
            print('If this step takes too long, you may specify --vis_fast = True.')

    def loop():
        n = 0
        while True:
            yield n
            n += 1

    #...............................................................................................
    # DRIVER:
    mode, input_source = check_input()
    video_fn = os.path.splitext(os.path.basename(input_source))

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader
    if mode == 'webcam':
        det_loader = WebCamDetectionLoader(input_source, get_detector(args), cfg, args)
        det_worker = det_loader.start()
    elif mode == 'detfile':
        det_loader = FileDetectionLoader(input_source, cfg, args)
        det_worker = det_loader.start()
    else:
        det_loader = DetectionLoader( input_source, get_detector(args), cfg, args,
                                      batchSize=args.detbatch, mode=mode, queueSize=args.qsize )
        det_worker = det_loader.start()

    # Load pose model
    pose_model = builder.build_sppe( cfg.MODEL, preset_cfg=cfg.DATA_PRESET,
                                     verbosity=args.verbosity )

    if args.verbosity==2:
        print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if args.pose_track:
        tracker = Tracker(tcfg, args)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    queueSize = 2 if mode == 'webcam' else args.qsize
    if args.save_video and mode != 'image':
        from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt

        if mode == 'video':
            video_save_opt['savepath'] = os.path.join( args.visoutpath, 'AlphaPose_'
                                                       + video_fn[0] + args.suffix + video_fn[1] )
        else:
            video_save_opt['savepath'] = os.path.join( args.visoutpath, 'AlphaPose_cam' +
                                                       str(input_source) + args.suffix + '.mp4' )
        video_save_opt.update(det_loader.videoinfo)
        writer = DataWriter( cfg, args, save_video=True, video_save_opt=video_save_opt,
                             queueSize=queueSize, video_fn_ne=video_fn[0] ).start()
    else:
        writer = DataWriter( cfg, args, save_video=False, queueSize=queueSize,
                             video_fn_ne=video_fn[0] ).start()

    if args.verbosity==0: disable_tqdm=True
    else disable_tqdm=False

    if mode == 'webcam':
        if args.verbosity:
            print('Webcam process initiated. Press Ctrl + C to terminate.')
        sys.stdout.flush()
        im_names_desc = tqdm(loop(), disable=disable_tqdm)
    else:
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True, disable=disable_tqdm)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                if args.pose_track:
                    boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                hm = hm.cpu() 
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                tqdm_str = 'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'
                im_names_desc.set_description( tqdm_str.format( dt=np.mean(runtime_profile['dt']),
                                                                pt=np.mean(runtime_profile['pt']),
                                                                pn=np.mean(runtime_profile['pn']) ))

        if args.verbosity:
            print_finish_info()
            while(writer.running()):
                time.sleep(1.5)
                print('Rendering remaining ' + str(writer.count()) + ' images in the queue.', end='\r')
        writer.stop()
        det_loader.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        if args.verbosity:
            print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            if args.verbosity:
                while(writer.running()):
                    time.sleep(1.5)
                    print('Rendering remaining ' + str(writer.count()) + ' images in the queue.', end='\r')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()