import os
import time
import cv2
import json
import threading
import asyncio
import numpy as np
import pretty_errors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR

import tensorboardX
from torch.utils.tensorboard import SummaryWriter

from models.model import ViTPose
from train_and_infer_async import train_async, inference_image

from utils.top_down_eval import keypoints_from_heatmaps
from utils.visualization import draw_points_and_skeleton, joints_dict

from utils.coco_utils import COCO_json_template
# import configs.ViTPose_base_IAML as student_cfg
# import configs.ViTPose_huge_coco_256x192 as teacher_cfg
import configs.ViTPose_small_coco_256x192 as student_cfg

from datasets.COCO import COCODataset
from torch.utils.data import Dataset, DataLoader
from models.losses import JointsMSELoss
from models.optimizer import LayerDecayOptimizer

def tensor_to_opencv(tensor, h=256, w=192):
    # convert tensor to opencv image
    # tensor: (1, 3, h, w)
    # return: (h, w, 3)
    if len(tensor.shape) > 3:
        tensor = tensor.squeeze(0)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    # permute array to hwc
    image = np.array(tensor).transpose(1, 2, 0)
    # adjust channel order to opencv and handle grayscale images
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # resize to hw
    image = cv2.resize(image, (w, h))
    # normalize the min and max values to 0-255
    image = image - image.min()
    try:
        image = image / image.max()
    except ZeroDivisionError:
        print(f"ZeroDivisionError: image.max() = {image.max()}, image.min() = {image.min()}")
        pass
    image = (image * 255).astype(np.uint8)
    return image

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

def main():

    CUR_PATH = os.path.dirname(__file__)
    teacher_checkpoint = f"./vitpose-b-multi-coco.pth"
    student_checkpoint = f"./vitpose_small.pth"

    # logging and visualization
    writer = SummaryWriter()

    # work with Mac GPU mps
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # device = torch.device("cpu")
    print(f"Using device: {device}")


    # Set dataset
    dataset = COCODataset(root_path=student_cfg.data_root, 
                           data_version=student_cfg.data_version,
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

    dataloader = DataLoader(dataset, batch_size=student_cfg.data['samples_per_gpu'], num_workers=0, pin_memory=False, shuffle=True, drop_last=False)

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True
    
    # init distributed env first, since logger depends on the dist info.
    distributed = False

    cv2.namedWindow('window', cv2.WINDOW_AUTOSIZE)

    # Create and load the model
    model_student = ViTPose(student_cfg.model)
    # model_student.load_state_dict(torch.load(student_checkpoint)['state_dict'])
    # model_student.keypoint_head.init_weights()

    # set part of the model to be trainable, set those parameters to zero, and freeze the rest
    # for name, param in model_student.named_parameters():
    #     # print(name, param.shape)
    #     if 'keypoint_head.final_layer' in name:
    #         # print(param)
    #         param.requires_grad = True
    #         # randomly initialize these parameters not to be zero
    #         param.data = torch.randn(param.shape)
    #     else:
    #         # freeze the rest
    #         param.requires_grad = False

    # torch.save({'state_dict': model_student.state_dict()}, student_cfg.checkpoint)
    model_student.to(device)
    model_student.train()
    # model_student.eval()
    
    # Loss function
    criterion = JointsMSELoss(use_target_weight=student_cfg.model['keypoint_head']['loss_keypoint']['use_target_weight'])
    
    # Optimizer
    lr = student_cfg.optimizer['lr']
    optimizer = AdamW(model_student.parameters(), lr=lr, betas=student_cfg.optimizer['betas'], weight_decay=student_cfg.optimizer['weight_decay'])
    
    # Layer-wise learning rate decay
    # lr_mult = [student_cfg.optimizer['paramwise_cfg']['layer_decay_rate']] * student_cfg.optimizer['paramwise_cfg']['num_layers']
    # layerwise_optimizer = LayerDecayOptimizer(optimizer, lr_mult)
    
    # ToDo: consider the proper learning rate and milestones

    # Learning rate scheduler (MultiStepLR)
    milestones = student_cfg.lr_config['step']
    gamma = 0.1
    scheduler = MultiStepLR(optimizer, milestones, gamma)

    # Warm-up scheduler
    # num_warmup_steps = student_cfg.lr_config['warmup_iters']  # Number of warm-up steps
    # warmup_factor = student_cfg.lr_config['warmup_ratio']  # Initial learning rate = warmup_factor * learning_rate
    # warmup_scheduler = LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: warmup_factor + (1.0 - warmup_factor) * step / num_warmup_steps
    # )
    
    global_step = 0

    for epoch in range(student_cfg.total_epochs):
        start_time = time.time()
        epoch_loss = []

        # print the current learning rate
        for param_group in optimizer.param_groups:
            epoch_lr = param_group['lr']
            # print(f"current learning rate: {param_group['lr']}")
        # write to tensorboard
        writer.add_scalar('learning_rate', epoch_lr, epoch)

        # for batch_idx, batch in dataloader:
        for batch_idx, batch in enumerate(dataloader):
            # layerwise_optimizer.zero_grad()
            optimizer.zero_grad()
            
            images_opencv, images, targets, target_weights, __ = batch

            images = images.to(device)
            targets = targets.to(device)
            target_weights = target_weights.to(device)
            # make target_weights to 1
            # target_weights = torch.ones_like(target_weights)

            outputs = model_student(images)

            ### debug code ###
            # make a temp debug dir to save images and targets
            # if not os.path.exists('./debug'):
            #     os.mkdir('./debug')
            
            # temp_size = min(10, images.size(0))
            # for i in range(temp_size):
            #     # convert images[1] to numpy array and save as image
            #     image = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
            #     image = (image - image.min()) / (image.max() - image.min()) * 255
            #     cv2.imwrite(f'./debug/{i}.jpg', image)

            #     # convert targets[1] to numpy array and save as image
            #     # combine all 17 targets into one image
            #     target = np.sum(targets[i].cpu().numpy(), axis=0)
            #     # resize target to the same size as image
            #     target = cv2.resize(target, (192, 256))
            #     target = (target - target.min()) / (target.max() - target.min()) * 255
            #     cv2.imwrite(f'./debug/{i}_target.jpg', target)

            #     # visualize the outputs as well
            #     # convert outputs[1] to numpy array and save as image
            #     # combine all 17 outputs into one image
            #     output = np.sum(outputs[i].cpu().detach().numpy(), axis=0)
            #     # resize output to the same size as image
            #     output = cv2.resize(output, (192, 256))
            #     output = (output - output.min()) / (output.max() - output.min()) * 255
            #     cv2.imwrite(f'./debug/{i}_output.jpg', output)

            loss = criterion(outputs, targets, target_weights) # if use_target_weight=True, then criterion(outputs, targets, target_weights)
            loss.backward()
            clip_grad_norm_(model_student.parameters(), ** student_cfg.optimizer_config['grad_clip'])
            
            # layerwise_optimizer.step()
            optimizer.step()

            # add loss to epoch_loss
            epoch_loss.append(loss.item())
            
            # print global step every n steps
            if global_step % 10 == 0:

                # loss_N_avg = round(sum(epoch_loss[-len(batch[0]):]) / len(batch[0]), 5)
                batch_loss = round(epoch_loss[-1], 5)
                # print(f'Batch: {batch_idx}, Loss: {batch_loss}, Global step: {global_step}')
                # write to tensorboard
                writer.add_scalar(f'Batch_loss', batch_loss, global_step)

                # reverse the tensors for tensorboard visualization
                image_preview = images_opencv[0].cpu().numpy()
                # convert Pillow image to opencv image
                image_preview = cv2.cvtColor(image_preview, cv2.COLOR_RGB2BGR)
                targets_preview = targets[0].cpu().numpy()
                outputs_heatmap = outputs[0].cpu().detach().numpy()

                # normalize the image back to opencv format
                # images_np = np.transpose(images[0].cpu().numpy(), (1, 2, 0))
                # images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min()) * 255
                # images_np = images_np.astype(np.uint8)
                org_h, org_w, _ = image_preview.shape

                # save an image with keypoints to tensorboard
                heatmaps = outputs[:1].detach().cpu().numpy() # N, 17, h/4, w/4

                points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]), unbiased=True, use_udp=True)

                points_confidence = np.concatenate([points[:, :, ::-1], prob], axis=2)

                # collect all frames, draw the keypoints and save the video
                for pid, point in enumerate(points_confidence):
                    # convert images[0] to numpy array
                    img_keypoints = draw_points_and_skeleton(image_preview.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid, points_color_palette='gist_rainbow', skeleton_color_palette='jet', points_palette_samples=10, confidence_threshold=0.4)

                # process the targets the same way as outputs for visualization
                target_weight_array = target_weights[0].cpu().numpy()
                # reshape the target_weight_array to (17, 1, 1) to multiply with targets_preview
                target_weight_array = np.reshape(target_weight_array, (17, 1, 1))

                # multiply the target_weight_array with targets_preview to downweight the dense head keypoints
                targets_preview = targets_preview * target_weight_array
                target_heatmap = np.sum(targets_preview, axis=0, keepdims=True)
                target_heatmap = tensor_to_opencv(target_heatmap)

                # combine all 17 outputs into one image but maintain the channel dimension
                outputs_heatmap = np.sum(outputs_heatmap, axis=0, keepdims=True)
                outputs_heatmap = tensor_to_opencv(outputs_heatmap)

                # horizontal stack the three array imgs img_keypoints, target_heatmap, outputs_heatmap into one image
                keypoints_and_heatmaps = np.hstack((img_keypoints, target_heatmap, outputs_heatmap))
                
                # also save the image to disk
                os.makedirs('./debug', exist_ok=True)
                cv2.imwrite(f'./debug/{global_step}.jpg', keypoints_and_heatmaps)
                # show the image in opencv window
                cv2.imshow('keypoints', keypoints_and_heatmaps)

                # before sending to tensorboard, permute the array to (C, H, W)
                # revert the RGB to BGR for tensorboard visualization
                keypoints_and_heatmaps = keypoints_and_heatmaps[:, :, ::-1]
                keypoints_and_heatmaps = np.transpose(keypoints_and_heatmaps, (2, 0, 1))
                writer.add_image('keypoints', keypoints_and_heatmaps, global_step)

            # end of an iteration
            # if global_step < num_warmup_steps:
            #     warmup_scheduler.step()
            global_step += 1

        # print time used for this epoch
        print(f'Epoch {epoch}, used {round(time.time() - start_time)} seconds to train on {len(dataset)} images')

        # print epoch loss
        epoch_loss_avg = round(sum(epoch_loss) / len(epoch_loss), 7)
        print(f'Epoch loss: {epoch_loss_avg}')
        # write to tensorboard
        # writer.add_scalar('Loss/epoch_avg', epoch_loss_avg, epoch)

        scheduler.step()

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == '__main__':
    main()