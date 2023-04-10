import argparse
import os
import json
from time import time

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
from utils.coco_utils import COCO_json_template

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

    anno_json = COCO_json_template
    anno_json['images'] = []
    anno_json['annotations'] = []

    # create the dataset directory to save the frames and the json file
    dataset_dir = vid_path.replace(".mp4", "")
    os.makedirs(dataset_dir, exist_ok=True)
    
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
        print(f">>> Processing frame {frame_count} ...", end='\r')
        if not ret or frame_count > stop_frame:
            break

        # img = Image.fromarray(frame)
        # img = frame

        imgPath = f"{dataset_dir}/{frame_count:012d}.jpg"
        if save_result:
            cv2.imwrite(imgPath, frame)

        # img = Image.open(vid_path)
        
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((img_size[1], img_size[0])),
            ])
        
        img_tensor = img_transform(frame).unsqueeze(0).to(device)
        
        # Feed to model
        # the VitPose model returns a list of 17 heatmaps, for 17 keypoints
        heatmaps = vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4

        # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]), unbiased=True, use_udp=True)
        points_confidence = np.concatenate([points[:, :, ::-1], prob], axis=2)

        breakpoint()

        # post process the keypoints by adding COCO visibility flag, the output is 17*3 (x1, y1, v1, ...)
        # https://cocodataset.org/#format-results
        points_coco = np.concatenate([points, np.ones((points.shape[0], points.shape[1], 1))], axis=2)
        # round the keypoints with 2 decimal places
        points_coco = np.round(points_coco, 2).reshape(-1).tolist()

        # convert ViTpose pose predictions to a json file in COCO format
        # https://cocodataset.org/#format-results
        img_info = {
            "license": 1,
            "id": frame_count,
            "file_name": f"{frame_count:012d}.jpg",
            "width": org_w,
            "height": org_h,
            # "date_captured": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        anno_info = {
            "image_id": frame_count,
            "iscrowd": 0,
            "category_id": 1,
            "num_keypoints": len(points[0]),
            "keypoints": points_coco,
            "score": 1,
            "id": frame_count,
            "bbox": [0, 0, org_w, org_h],
            "area": org_w*org_h,
        }
        anno_json['images'].append(img_info)
        anno_json['annotations'].append(anno_info)


        # breakpoint()

        # collect all frames, draw the keypoints and save the video
        for pid, point in enumerate(points_confidence):
            img = np.array(img) # Pillow read img as RGB, cv2 read img as BGR
            # (Pdb) img.shape (1440, 1080, 3)
            # (Pdb) points_confidence.shape (1, 17, 3)
            # (Pdb) point.shape (17, 3)

            #[:, :, ::-1] # RGB to BGR for cv2 modules
            img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                           points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                           points_palette_samples=10, confidence_threshold=0.4)
        
        all_frames.append(img)
        all_points.append(points)
        frame_count += 1

    if save_result:
        # save the dataset json
        with open(f"{dataset_dir}/dataset.json", "w") as f:
            json.dump(anno_json, f)
        print(f">>> Save COCO style json to {dataset_dir}/dataset.json")

        # save the video
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
    # from configs.ViTPose_base_coco_256x192 import model as model_cfg
    # from configs.ViTPose_base_coco_256x192 import data_cfg

    from configs.ViTPose_huge_coco_256x192 import model as teacher_cfg
    from configs.ViTPose_huge_coco_256x192 import data_cfg

    # work with Mac GPU mps
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', nargs='+', type=str, default='examples/baby_shark_pinkfong_vertical.mp4', help='video path')
    args = parser.parse_args()
    
    CUR_DIR = os.path.dirname(__file__)
    # CKPT_PATH = f"{CUR_DIR}/vitpose-b-multi-coco.pth"
    CKPT_PATH = f"{CUR_DIR}/vitpose-h-multi-coco.pth"
    
    img_size = data_cfg['image_size']
    t0, t1 = 0, 20

    print(args.video_path)
    keypoints, frames = inference_video(vid_path=args.video_path, start_t=t0, stop_t=t1, img_size=img_size, model_cfg=teacher_cfg, ckpt_path=CKPT_PATH, device=device, save_result=True)