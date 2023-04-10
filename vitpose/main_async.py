import os
import time
import cv2
import json
import threading
import asyncio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import tensorboardX
from tensorboardX import SummaryWriter


from models.model import ViTPose
from train_and_infer_async import train_async, inference_image

from utils.coco_utils import COCO_json_template
import configs.ViTPose_base_IAML as student_cfg
import configs.ViTPose_huge_coco_256x192 as teacher_cfg

from imutils.video import WebcamVideoStream
from imutils.video import FPS


def read_calibration_video(video_path):
    # read every frame from the video at vid_path
    vid = cv2.VideoCapture(video_path)
    vid_frames = []
    if not vid.isOpened(): raise IOError(f"Couldn't open video: {video_path}")

    # get the image size of the video
    org_w, org_h = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get framerate of the video
    fps = vid.get(cv2.CAP_PROP_FPS)

    while True:
        vid_ret, vid_frame = vid.read()
        if not vid_ret: break
        vid_frames.append(vid_frame)

    return vid_frames, org_w, org_h, fps

async def main():

    CUR_PATH = os.path.dirname(__file__)
    teacher_checkpoint = f"./vitpose-b-multi-coco.pth"
    video_path = 'examples/baby_shark_pinkfong_vertical.mp4'

    # logging and visualization
    writer = SummaryWriter()

    # temporary dataset directory to save the frames and the json file
    dataset_dir = student_cfg.data_root
    dataset_imgs = f"{dataset_dir}/val2017"
    dataset_json = student_cfg.data_json

    # purge the dataset directory
    if os.path.exists(dataset_dir):
        os.system(f"rm -rf {dataset_dir}")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(dataset_imgs, exist_ok=True)
    os.makedirs(f"{dataset_dir}/annotations/", exist_ok=True)

    teacher_cfg = student_cfg

    # work with Mac GPU mps
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # build a COCO style dataset from the webcam frames and teacher_model predictions
    anno_json = COCO_json_template
    anno_json['images'] = []
    anno_json['annotations'] = []

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True
    
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

    # Read the calibration video
    vid_frames, org_w, org_h, fps= read_calibration_video(video_path)
    vid_frameid = 0

    cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)

    # mp.set_start_method('fork', force=True)
    # mp.set_start_method('fork')
    # pool = mp.Pool(processes=2)
    # lock = threading.Lock()

    # Pyhton multiprocessing to train and inference at the same time
    # mp.set_start_method('spawn', force=True)
    interval = 5
    
    # Create and load the model
    model_student = ViTPose(student_cfg.model)
    # save the new model to cfg.checkpoint first
    model_student.load_state_dict(torch.load(teacher_checkpoint)['state_dict'])
    # model_student.keypoint_head.init_weights()

    # set part of the model to be trainable, set those parameters to zero, and freeze the rest
    for name, param in model_student.named_parameters():
        # print(name, param.shape)
        if 'keypoint_head.final_layer' in name:
            # print(param)
            param.requires_grad = True
            # randomly initialize these parameters not to be zero
            param.data = torch.randn(param.shape)
        else:
            # freeze the rest
            param.requires_grad = False

    # torch.save({'state_dict': model_student.state_dict()}, student_cfg.checkpoint)
    model_student.to(device)
    # model_student.eval()
    
    model_teacher = ViTPose(teacher_cfg.model)
    model_teacher.load_state_dict(torch.load(teacher_checkpoint)['state_dict'])
    model_teacher.to(device)
    model_teacher.eval()

    # Open the webcam
    # webcam = cv2.VideoCapture(0)
    webcam = WebcamVideoStream(src=0).start()
    cam_frameid = 0
    fps = FPS().start()
    fps_display = 0

    # confirm that the webcam is open before starting the processes
    # cam_ret, cam_frame = webcam.read()
    # if not cam_ret: raise IOError(f"Couldn't open webcam")
    cam_frame = webcam.read()

    # train_task = None
    train_task = asyncio.create_task(train_async(device, cam_frameid))

    while cam_frame is not None:
        # cam_ret, cam_frame = webcam.read()
        cam_frame = webcam.read()
        vid_frame = vid_frames[vid_frameid]

        # the webcam frame is larger and wider than the video frame
        # first resize the webcam frame to height org_h but maintain its ratio
        # then crop the webcam frame to the same width as the video frame
        cam_frame = cv2.resize(cam_frame, (int(org_h*cam_frame.shape[1]/cam_frame.shape[0]), org_h))
        # crop webcam frame to org_w
        cam_frame = cam_frame[:, int((cam_frame.shape[1]-org_w)/2):int((cam_frame.shape[1]+org_w)/2)]
        # mirror the webcam frame
        cam_frame = cv2.flip(cam_frame, 1)
        
        # debug
        cam_frame = vid_frame

        # cam_keypoints_img, _ = await inference_image(model_student, cam_frame, device, None, None, False)
        cam_keypoints_img, _ = await inference_image(model_student, cam_frame, device, None, None, False)

        vid_keypoints_img, anno_json = await inference_image(model_teacher, vid_frame, device, anno_json, cam_frameid, save_json=True)

        # print(f">>> Processing frame {vid_frameid} ...", end='\r')

        # save the current camera to dataset folder as the training data
        file_name = os.path.join(dataset_imgs, f"{cam_frameid:012d}.jpg")
        cv2.imwrite(file_name, cam_frame)

        # save dataset json file every 100 frames
        if cam_frameid > 1 and cam_frameid % 200 == 0:
            with open(dataset_json, 'w') as f:
                json.dump(anno_json, f)
            # print a message and show how many frames have been collected so far
            print(f">>> {cam_frameid} frames collected")

            await asyncio.gather(train_task)

            if train_task.done():
                # train_result = train_task.result()

                # check if model file exists
                if os.path.isfile(student_cfg.checkpoint):
                    print(f'now loading new model weights, is training finished?')
                    model_student.load_state_dict(torch.load(student_cfg.checkpoint))

                train_task = asyncio.create_task(train_async(device, cam_frameid))

            # train_return = pool.apply_async(train_async, (device, cam_frameid, train_ongoing)).get()
            # train_thread = threading.Thread(target=train_async, args=(device, cam_frameid))
            # train_thread.start()

        # draw the fps_display on the img with bold and large font size
        cv2.putText(cam_keypoints_img, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # time.sleep(f_interval)

        # combine the original image and the image with keypoints
        frame_img = cv2.hconcat([cam_keypoints_img, vid_keypoints_img])

        # pop a window to show the image and refresh the window
        cv2.imshow('window', frame_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # do a bit of cleanup
            cv2.destroyAllWindows()
            webcam.stop()
            break

        cam_frameid += 1
        vid_frameid += 1
        fps.update()

        if vid_frameid == len(vid_frames):
            vid_frameid = 0

        if vid_frameid % 100 == 0:
            fps.stop()
            fps_display = round(fps.fps(), 2)
            print(f"FPS: {fps_display}")

if __name__ == '__main__':
    asyncio.run(main())