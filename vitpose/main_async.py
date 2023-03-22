import os
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import ViTPose
from train_and_infer_async import train_process, inference_image

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
    # calculate per frame interval based on the framerate
    f_interval = 0.1 # 1/fps
    frame_count = 0

    while True:
        vid_ret, vid_frame = vid.read()
        if not vid_ret: break
        vid_frames.append(vid_frame)

    return vid_frames, org_w, org_h, fps, f_interval, frame_count

if __name__ == '__main__':

    CUR_PATH = os.path.dirname(__file__)
    student_checkpoint = f"./vitpose-b-multi-coco.pth"
    teacher_checkpoint = f"./vitpose-h-multi-coco.pth"
    video_path = 'examples/baby_shark_pinkfong_vertical.mp4'

    teacher_cfg = student_cfg
    teacher_checkpoint = student_checkpoint

    # work with Mac GPU mps
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # device = torch.device("cpu")
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

    # Read the calibration video
    vid_frames, org_w, org_h, fps, f_interval, frame_count = read_calibration_video(video_path)

    cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)

    # Pyhton multiprocessing to train and inference at the same time
    # mp.set_start_method('spawn', force=True)
    interval = 5
    
    # Create and load the model
    model_student = ViTPose(student_cfg.model)
    # model.load_state_dict(torch.load(CKPT_PATH)['state_dict'])
    model_student.to(device)
    model_student.train()
    
    model_teacher = ViTPose(teacher_cfg.model)
    model_teacher.load_state_dict(torch.load(teacher_checkpoint)['state_dict'])
    model_teacher.to(device)
    model_teacher.eval()

    # Open the webcam
    # webcam = cv2.VideoCapture(0)
    webcam = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    fps_display = 0

    # confirm that the webcam is open before starting the processes
    # cam_ret, cam_frame = webcam.read()
    # if not cam_ret: raise IOError(f"Couldn't open webcam")
    cam_frame = webcam.read()

    while cam_frame is not None:
        # cam_ret, cam_frame = webcam.read()
        cam_frame = webcam.read()
        vid_frame = vid_frames[frame_count]

        # the webcam frame is larger and wider than the video frame
        # first resize the webcam frame to height org_h but maintain its ratio
        # then crop the webcam frame to the same width as the video frame
        cam_frame = cv2.resize(cam_frame, (int(org_h*cam_frame.shape[1]/cam_frame.shape[0]), org_h))
        # crop webcam frame to org_w
        cam_frame = cam_frame[:, int((cam_frame.shape[1]-org_w)/2):int((cam_frame.shape[1]+org_w)/2)]
        # mirror the webcam frame
        cam_frame = cv2.flip(cam_frame, 1)

        cam_keypoints_img = inference_image(model_student, cam_frame, device)
        vid_keypoints_img = inference_image(model_teacher, vid_frame, device)

        print(f">>> Processing frame {frame_count} ...", end='\r')

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

        frame_count += 1
        fps.update()

        if frame_count == len(vid_frames):
            frame_count = 0

        if frame_count % 100 == 0:
            fps.stop()
            fps_display = round(fps.fps(), 2)
            print(f"FPS: {fps_display}")