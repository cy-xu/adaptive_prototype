import os
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from models.model import ViTPose
from train_and_infer import train_process, inference_video
import configs.ViTPose_base_IAML as cfg


if __name__ == '__main__':

    CUR_PATH = os.path.dirname(__file__)
    CKPT_PATH = f"./vitpose-b-multi-coco.pth"
    video_path = 'examples/baby_shark_pinkfong_vertical.mp4'

    # work with Mac GPU mps
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # cfg = b_iaml_cfg
    
    # set cudnn_benchmark
    # torch.backends.cudnn.benchmark = True
    
    # # Set work directory
    # cfg.__setattr__('work_dir', f"{CUR_PATH}/runs/train")
    # if not os.path.exists(cfg.work_dir):
    #     os.makedirs(cfg.work_dir)

    # if cfg.optimizer['autoscale_lr']:
    #     # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
    #     cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # # init the logger before other steps
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file)

    # # init the meta dict to record some important information such as
    # # environment info and seed, which will be logged
    # meta = dict()

    # # log some basic info
    # logger.info(f'Distributed training: {distributed}')

    # # set random seeds
    # seed = init_random_seed(cfg.seed, device=device)
    # logger.info(f"Set random seed to {seed}, "
    #             f"deterministic: {cfg.deterministic}")
    # set_random_seed(seed, deterministic=cfg.deterministic)
    # meta['seed'] = seed



    # shared_model = mp.Array('d', [0.0]*model_params)

    # Pyhton multiprocessing to train and inference at the same time
    mp.set_start_method('spawn', force=True)
    event = mp.Event()
    interval = 5
    vid_queue = Queue()
    cam_queue = Queue()
    
    # Create and load the model
    model = ViTPose(cfg.model)
    model.load_state_dict(torch.load(CKPT_PATH)['state_dict'])
    model.train()

    # Create a shared memory buffer to hold the model parameters
    # model = Net().to(device)
    model.share_memory()

    processes = []
    p_inference = mp.Process(target=inference_video, args=(model, vid_queue, cam_queue, device))
    p_training = mp.Process(target=train_process, args=(model, device, event, interval))

    # Start the inference and training processes
    # p_inference = mp.Process(target=inference_video, args=(model, video_path, cfg))
    # p_training = mp.Process(target=train_process, args=(model, cfg))

    # read every frame from the video at vid_path
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened(): raise IOError(f"Couldn't open video: {video_path}")

    # get the image size of the video
    org_w, org_h = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get framerate of the video
    fps = vid.get(cv2.CAP_PROP_FPS)
    # calculate per frame interval based on the framerate
    f_interval = 0.1 # 1/fps
    frame_count = 0

    # Open the webcam
    webcam = cv2.VideoCapture(0)

    # confirm that the webcam is open before starting the processes
    cam_ret, cam_frame = webcam.read()
    if not cam_ret: raise IOError(f"Couldn't open webcam")

    p_training.start()
    processes.append(p_training)

    p_inference.start()
    processes.append(p_inference)

    while True:
        vid_ret, vid_frame = vid.read()
        cam_ret, cam_frame = webcam.read()

        # the webcam frame is larger and wider than the video frame
        # first resize the webcam frame to height org_h but maintain its ratio
        # then crop the webcam frame to the same width as the video frame
        if cam_ret:
            # resize webcam frame to org_h
            cam_frame = cv2.resize(cam_frame, (int(org_h*cam_frame.shape[1]/cam_frame.shape[0]), org_h))
            # crop webcam frame to org_w
            cam_frame = cam_frame[:, int((cam_frame.shape[1]-org_w)/2):int((cam_frame.shape[1]+org_w)/2)]
            
        print(f">>> Processing frame {frame_count} ...", end='\r')
        if not (vid_ret and cam_ret): break

        vid_queue.put(vid_frame)
        cam_queue.put(cam_frame)
        time.sleep(f_interval)

        frame_count += 1

    vid_queue.put(None)

    for p in processes:
        p.join()