import os
import time
import torch
import torch.nn as nn
import numpy as np
import cv2

from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from torchvision.transforms import transforms

from models.losses import JointsMSELoss
from models.optimizer import LayerDecayOptimizer
from datasets.COCO import COCODataset
import configs.ViTPose_base_IAML as iaml_cfg

from utils.util import init_random_seed, set_random_seed
from utils.dist_util import get_dist_info, init_dist
from utils.logging import get_root_logger

from utils.top_down_eval import keypoints_from_heatmaps
from utils.visualization import draw_points_and_skeleton, joints_dict

def train_process(model, device, event, interval, lock):
    print(f">>> Entered train_process function")

    cfg = iaml_cfg

    # Set dataset
    dataset = COCODataset(root_path=cfg.data_root, 
                           data_version=cfg.data_version,
                           is_train=True, 
                           use_gt_bboxes=True,
                           image_width=192, 
                           image_height=256,
                           scale=True, 
                           scale_factor=0.35, 
                           flip_prob=0.5, 
                           rotate_prob=0.5, 
                           rotation_factor=45., 
                           half_body_prob=0.3,
                           use_different_joints_weight=True, 
                           heatmap_sigma=3, 
                           soft_nms=False)
    
    # logger = get_root_logger()
    
    # Prepare data loaders
    # datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]

    # ToDo: in early stage of IAML, batch_size should adapt to dataset size

    dataloader = DataLoader(dataset, batch_size=cfg.data['samples_per_gpu'], num_workers=cfg.data['workers_per_gpu'], pin_memory=True)
    
    # Loss function
    criterion = JointsMSELoss(use_target_weight=cfg.model['keypoint_head']['loss_keypoint']['use_target_weight'])
    
    # Optimizer
    lr = cfg.optimizer['lr']
    optimizer = AdamW(model.parameters(), lr=lr, betas=cfg.optimizer['betas'], weight_decay=cfg.optimizer['weight_decay'])
    
    # Layer-wise learning rate decay
    lr_mult = [cfg.optimizer['paramwise_cfg']['layer_decay_rate']] * cfg.optimizer['paramwise_cfg']['num_layers']
    layerwise_optimizer = LayerDecayOptimizer(optimizer, lr_mult)
    
    # ToDo: consider the proper learning rate and milestones

    # Learning rate scheduler (MultiStepLR)
    milestones = cfg.lr_config['step']
    gamma = 0.1
    scheduler = MultiStepLR(optimizer, milestones, gamma)

    # Warm-up scheduler
    num_warmup_steps = cfg.lr_config['warmup_iters']  # Number of warm-up steps
    warmup_factor = cfg.lr_config['warmup_ratio']  # Initial learning rate = warmup_factor * learning_rate
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_factor + (1.0 - warmup_factor) * step / num_warmup_steps
    )
    
    global_step = 0

    # actual training loop
    # ToDo: this should by dynamic 

    # wait for the event to start training
    # while not event.wait(interval):
    while global_step < 1000:

        for epoch in range(cfg.total_epochs):
            print(f'Epoch {epoch}')
            epoch_loss = []

            # lock.acquire()

            # for batch_idx, batch in dataloader:
            for batch_idx, batch in enumerate(dataloader):
                layerwise_optimizer.zero_grad()
                
                images, targets, target_weights, __ = batch
                images = images.to(device)
                targets = targets.to(device)
                target_weights = target_weights.to(device)

                outputs = model(images)
                
                loss = criterion(outputs, targets, target_weights) # if use_target_weight=True, then criterion(outputs, targets, target_weights)
                loss.backward()
                clip_grad_norm_(model.parameters(), **cfg.optimizer_config['grad_clip'])
                layerwise_optimizer.step()

                # add loss to epoch_loss
                epoch_loss.append(loss.item())
                
                # print global step every 10 steps
                pid = os.getpid()
                if global_step % 3 == 0:
                    print(f'Batch: {batch_idx}, Loss: {round(loss.item(), 8)}, Global step: {global_step}, PID: {pid}')

                if global_step < num_warmup_steps:
                    warmup_scheduler.step()
                global_step += 1

            # print epoch loss
            print(f'Epoch loss: {sum(epoch_loss) / len(epoch_loss)}')

            scheduler.step()
            
            # Write the new state of the model to shared memory
            # model[:] = model.state_dict()

            # lock.release()

    # fp16 setting
    # ToDo: we should implement fp16 training
    
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     raise NotImplementedError()
    

# @torch.no_grad()
def inference_video(model, vid_queue, cam_queue, device, lock):
    print(f">>> Entered inference_video function")

    # Create a window to show both videos
    # cv2.namedWindow('Double Video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)
    # cv2.CV_WINDOW_AUTOSIZE

    cfg = iaml_cfg.data_cfg

    img_size = cfg['image_size']
    img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((img_size[1], img_size[0])),
    ])

    # Initialize the position and size of the window
    # cv2.moveWindow('Double Video', 0, 0)
    # cv2.resizeWindow('Double Video', org_w*2, org_h)

    while True:

        if not vid_queue.empty():
            vid_frame = vid_queue.get()

        if not cam_queue.empty():
            cam_frame = cam_queue.get()
            org_h, org_w, _ = cam_frame.shape

        # if frame is None:
        #     break

        # org_h, org_w, _ = vid_frame.shape

        # img_size = cfg['image_size']
        # print(f">>> Original image size: {org_h} X {org_w} (height X width)")
        # print(f">>> Resized image size: {img_size[1]} X {img_size[0]} (height X width)")
        # print(f">>> Scale change: {org_h/img_size[1]}, {org_w/img_size[0]}")

        #  a for loop to read every frame from the video
        # tic = time()
        # start_frame = int(start_t * fps)
        # frame_count = start_frame
        # stop_frame = int(stop_t * fps)
        # vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # lock.acquire()

        # Feed to model
        # the VitPose model returns a list of 17 heatmaps, for 17 keypoints
        # no grad
        with torch.no_grad():
            # img_tensor = img_transform(vid_frame).unsqueeze(0).to(device)
            img_tensor = img_transform(cam_frame).unsqueeze(0).to(device)

            heatmaps = model(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4

        # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]), unbiased=True, use_udp=True)
        points_confidence = np.concatenate([points[:, :, ::-1], prob], axis=2)

        # collect all frames, draw the keypoints and save the video
        for pid, point in enumerate(points_confidence):
            # img = np.array(img) # Pillow read img as RGB, cv2 read img as BGR
            # (Pdb) img.shape (1440, 1080, 3)
            # (Pdb) points_confidence.shape (1, 17, 3)
            # (Pdb) point.shape (17, 3)

            #[:, :, ::-1] # RGB to BGR for cv2 modules
            img_keypoints = draw_points_and_skeleton(cam_frame.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                            points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                            points_palette_samples=10, confidence_threshold=0.4)

        # combine the original image and the image with keypoints
        frame_img = cv2.hconcat([img_keypoints, vid_frame])

        pid = os.getpid()
        # print(f'>>> Process {pid} is processing frame {frame_count} ...', end='\r')

        # pop a window to show the image and refresh the window
        cv2.imshow('window', frame_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # lock.release()

    cv2.destroyAllWindows()

    # elapsed_time = time()-tic
    # print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")    


