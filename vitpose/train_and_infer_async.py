import os
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import asyncio

from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from torchvision.transforms import transforms

from models.model import ViTPose
from models.losses import JointsMSELoss
from models.optimizer import LayerDecayOptimizer
from datasets.COCO import COCODataset
import configs.ViTPose_base_IAML as iaml_cfg

from utils.util import init_random_seed, set_random_seed
from utils.dist_util import get_dist_info, init_dist
from utils.logging import get_root_logger

from utils.top_down_eval import keypoints_from_heatmaps
from utils.visualization import draw_points_and_skeleton, joints_dict

async def train_async(device, frame_id):
    print(f"entered train_async")

    if frame_id == 0:
        return 1
    
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
                           rotate_prob=0., 
                           rotation_factor=0., 
                           half_body_prob=0.,
                           use_different_joints_weight=True, 
                           heatmap_sigma=3, 
                           soft_nms=False)
    
    # logger = get_root_logger()
    
    # Prepare data loaders
    # datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]

    # ToDo: in early stage of IAML, batch_size should adapt to dataset size

    model = ViTPose(cfg.model)
    # check if file cfg.checkpoint exists
    if os.path.isfile(cfg.checkpoint):
        model.load_state_dict(torch.load(cfg.checkpoint))

    model.to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=cfg.data['samples_per_gpu'], num_workers=cfg.data['workers_per_gpu'], pin_memory=True, shuffle=True, drop_last=True)
    
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

    # print the current learning rate
    for param_group in optimizer.param_groups:
        print(f"current learning rate: {param_group['lr']}")

    # actual training loop
    # ToDo: this should by dynamic 

    # wait for the event to start training
    # while not event.wait(interval):
    for epoch in range(cfg.total_epochs):
        start_time = time.time()
        epoch_loss = []
        print(f'Epoch {epoch} trains on {len(dataset)} images')

        # for batch_idx, batch in dataloader:
        for batch_idx, batch in enumerate(dataloader):
            layerwise_optimizer.zero_grad()
            
            images, targets, target_weights, __ = batch
            images = images.to(device)
            targets = targets.to(device)
            target_weights = target_weights.to(device)
            # make target_weights to 1
            target_weights = torch.ones_like(target_weights)

            outputs = model(images)

            ### debug code ###
            # make a temp debug dir to save images and targets
            if not os.path.exists('./debug'):
                os.mkdir('./debug')
            
            temp_size = min(10, images.size(0))
            for i in range(temp_size):
                # convert images[1] to numpy array and save as image
                image = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
                image = (image - image.min()) / (image.max() - image.min()) * 255
                cv2.imwrite(f'./debug/{i}.jpg', image)

                # convert targets[1] to numpy array and save as image
                # combine all 17 targets into one image
                target = np.sum(targets[i].cpu().numpy(), axis=0)
                # resize target to the same size as image
                target = cv2.resize(target, (192, 256))
                target = (target - target.min()) / (target.max() - target.min()) * 255
                cv2.imwrite(f'./debug/{i}_target.jpg', target)

                # visualize the outputs as well
                # convert outputs[1] to numpy array and save as image
                # combine all 17 outputs into one image
                output = np.sum(outputs[i].cpu().detach().numpy(), axis=0)
                # resize output to the same size as image
                output = cv2.resize(output, (192, 256))
                output = (output - output.min()) / (output.max() - output.min()) * 255
                cv2.imwrite(f'./debug/{i}_output.jpg', output)

            loss = criterion(outputs, targets, target_weights) # if use_target_weight=True, then criterion(outputs, targets, target_weights)
            loss.backward()
            clip_grad_norm_(model.parameters(), **cfg.optimizer_config['grad_clip'])
            layerwise_optimizer.step()

            # add loss to epoch_loss
            epoch_loss.append(loss.item())
            
            # print global step every 10 steps
            pid = os.getpid()
            if global_step % 2 == 0:
                print(f'Batch: {batch_idx}, Loss: {round(loss.item(), 8)}, Global step: {global_step}, PID: {pid}')

            if global_step < num_warmup_steps:
                warmup_scheduler.step()

            global_step += 1

        # print time used for this epoch
        print(f'Used {time.time() - start_time} to train on {frame_id} images')
        # print epoch loss
        print(f'Epoch loss: {sum(epoch_loss) / len(epoch_loss)}')

        scheduler.step()

    # save the model file to cfg.data_root with frame_id
    torch.save(model.state_dict(), cfg.checkpoint)

    # fp16 setting
    # ToDo: we should implement fp16 training
    
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     raise NotImplementedError()

    # release model from CPU and GPU memory
    del model
    del dataset
    # torch.cuda.empty_cache()

    # return model, train_ongoing
    # await asyncio.sleep(0)


# @torch.no_grad()
async def inference_image(model, img_frame, device, anno_json, frame_id, save_json=False):
    cfg = iaml_cfg.data_cfg

    img_size = cfg['image_size']
    img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((img_size[1], img_size[0]), antialias=True),
    ])

    # Initialize the position and size of the window
    # cv2.moveWindow('Double Video', 0, 0)
    # cv2.resizeWindow('Double Video', org_w*2, org_h)

    org_h, org_w, _ = img_frame.shape

    with torch.no_grad():
        # img_tensor = img_transform(vid_frame).unsqueeze(0).to(device)
        img_tensor = img_transform(img_frame).unsqueeze(0).to(device)

        heatmaps = model(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4

    # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]), unbiased=True, use_udp=True)

    points_confidence = np.concatenate([points[:, :, ::-1], prob], axis=2)

    if save_json:
        # post process the keypoints by adding COCO visibility flag, the output is 17*3 (x1, y1, v1, ...)
        # https://cocodataset.org/#format-results
        points_coco = np.concatenate([points, np.ones((points.shape[0], points.shape[1], 1))], axis=2)
        # round the keypoints with 2 decimal places
        points_coco = np.round(points_coco, 2).reshape(-1).tolist()

        # convert ViTpose pose predictions to a json file in COCO format
        # https://cocodataset.org/#format-results
        img_info = {
            "license": 1,
            "id": frame_id,
            "file_name": f"{frame_id:012d}.jpg",
            "width": org_w,
            "height": org_h,
            # "date_captured": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }

        anno_info = {
            "image_id": frame_id,
            "iscrowd": 0,
            "category_id": 1,
            "num_keypoints": len(points[0]),
            "keypoints": points_coco,
            "score": 1,
            "id": frame_id,
            "bbox": [org_w//2,org_h//2,org_w,org_h],
            "area": org_w*org_h,
        }

        anno_json['images'].append(img_info)
        anno_json['annotations'].append(anno_info)

    # collect all frames, draw the keypoints and save the video
    for pid, point in enumerate(points_confidence):
        # img = np.array(img) # Pillow read img as RGB, cv2 read img as BGR
        # (Pdb) img.shape (1440, 1080, 3)
        # (Pdb) points_confidence.shape (1, 17, 3)
        # (Pdb) point.shape (17, 3)

        #[:, :, ::-1] # RGB to BGR for cv2 modules
        img_keypoints = draw_points_and_skeleton(img_frame.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                        points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                        points_palette_samples=10, confidence_threshold=0.2)


        # pid = os.getpid()
        # print(f'>>> Process {pid} is processing frame {frame_count} ...', end='\r')


    # elapsed_time = time()-tic
    # print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time**-1: .1f} fps]\n")    

    return img_keypoints, anno_json