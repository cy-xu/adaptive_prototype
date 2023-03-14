import argparse
import os.path as osp

import torch
from torch import Tensor

from pathlib import Path
import cv2
import numpy as np

from time import time
from PIL import Image
from torchvision.transforms import transforms

from models.model import ViTPose
from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.dist_util import get_dist_info, init_dist
from utils.top_down_eval import keypoints_from_heatmaps

__all__ = ['inference_video']
            
            
@torch.no_grad()
def inference_video(vid_path: Path, start_t: int, stop_t: int, img_size: tuple[int, int],
              model_cfg: dict, ckpt_path: Path, device: torch.device, save_result: bool=True) -> np.ndarray:
    
    # Prepare model
    vit_pose = ViTPose(model_cfg)
    vit_pose.load_state_dict(torch.load(ckpt_path)['state_dict'])
    vit_pose.to(device)
    vit_pose.eval()

    all_frames = []
    all_points = []
    
    # Prepare input data
    # read every frame from the video at vid_path
    vid = cv2.VideoCapture(vid_path)
    if not vid.isOpened():
        raise IOError(f"Couldn't open video: {vid_path}")

    # get the image size of the video
    org_w, org_h = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get framerate of the video
    fps = vid.get(cv2.CAP_PROP_FPS)

    print(f">>> Original image size: {org_h} X {org_w} (height X width)")
    print(f">>> Resized image size: {img_size[1]} X {img_size[0]} (height X width)")
    print(f">>> Scale change: {org_h/img_size[1]}, {org_w/img_size[0]}")

    #  a for loop to read every frame from the video
    tic = time()
    start_frame = int(start_t * fps)
    frame_count = start_frame
    stop_frame = int(stop_t * fps)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ret, frame = vid.read()
        frame_count += 1
        print(f">>> Processing frame {frame_count} ...", end='\r')
        if not ret or frame_count > stop_frame:
            break

        img = Image.fromarray(frame)
        # img = Image.open(vid_path)
        img_tensor = transforms.Compose ([transforms.Resize((img_size[1], img_size[0])), transforms.ToTensor()])(img).unsqueeze(0).to(device)
        
        # Feed to model
        # the VitPose model returns a list of 17 heatmaps, for 17 keypoints
        heatmaps = vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4

        # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]),
                                            unbiased=True, use_udp=True)
        points = np.concatenate([points[:, :, ::-1], prob], axis=2)

        # collect all frames, draw the keypoints and save the video
        for pid, point in enumerate(points):
            img = np.array(img) # Pillow read img as RGB, cv2 read img as BGR
            #[:, :, ::-1] # RGB to BGR for cv2 modules
            img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                           points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                           points_palette_samples=10, confidence_threshold=0.4)
        
        all_frames.append(img)
        all_points.append(points)

    if save_result:
            save_name = vid_path.replace(".mp4", "_pose_result.mp4")
            out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (org_w, org_h))
            for frame in all_frames:
                out.write(frame)
            out.release()
            print(f">>> Save result video to {save_name}")
                

    elapsed_time = time()-tic
    print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")    
    
    return all_points, all_frames
    

if __name__ == "__main__":
    from configs.ViTPose_base_coco_256x192 import model as model_cfg
    from configs.ViTPose_base_coco_256x192 import data_cfg

    # work with Mac GPU mps
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', nargs='+', type=str, default='examples/vertical_chinese_calisthenics.mp4', help='video path')
    args = parser.parse_args()
    
    CUR_DIR = osp.dirname(__file__)
    CKPT_PATH = f"{CUR_DIR}/vitpose-b-multi-coco.pth"
    
    img_size = data_cfg['image_size']
    t0, t1 = 0, 20

    print(args.video_path)
    keypoints, frames = inference_video(vid_path=args.video_path, start_t=t0, stop_t=t1, img_size=img_size, model_cfg=model_cfg, ckpt_path=CKPT_PATH, device=device, save_result=True)