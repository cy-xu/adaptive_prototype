import torch
import torch.nn as nn
import numpy as np

from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from models.losses import JointsMSELoss
from models.optimizer import LayerDecayOptimizer

from utils.dist_util import get_dist_info, init_dist
from utils.logging import get_root_logger
from utils.top_down_eval import keypoints_from_heatmaps

# work with Mac GPU mps
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def train_model(model: nn.Module, datasets: Dataset, cfg: dict, distributed: bool, validate: bool,  timestamp: str, meta: dict) -> None:
    logger = get_root_logger()
    
    # Prepare data loaders
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]

    if distributed:
        samplers = [DistributedSampler(ds, num_replicas=len(cfg.gpu_ids), rank=torch.cuda.current_device(), shuffle=True, drop_last=False) for ds in datasets]
    else:
        samplers = [None for ds in datasets]

    # ToDo: in early stage of IAML, batch_size should adapt to dataset size

    dataloaders = [DataLoader(ds, batch_size=cfg.data['samples_per_gpu'], sampler=sampler, num_workers=cfg.data['workers_per_gpu'], pin_memory=False) for ds, sampler in zip(datasets, samplers)]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        model = DistributedDataParallel(model, 
                                        device_ids=[torch.cuda.current_device()], 
                                        broadcast_buffers=False, 
                                        find_unused_parameters=find_unused_parameters)
    else:
        # model = DataParallel(model, device_ids=cfg.gpu_ids)

        # send model to Mac GPU mps
        model = model.to(device)
    
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
    
    model.train()
    global_step = 0

    # actual training loop
    # ToDo: this should by dynamic 

    for dataloader in dataloaders:
        for epoch in range(cfg.total_epochs):
            print(f'Epoch {epoch}')
            epoch_loss = []

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
                
                if global_step < num_warmup_steps:
                    warmup_scheduler.step()
                global_step += 1

                # print global step every 10 steps
                if global_step % 10 == 0:
                    print(f'Global step: {global_step}')

                # debug visualization
                # the VitPose model returns a list of 17 heatmaps, for 17 keypoints
                heatmaps = outputs.detach().cpu().numpy() # N, 17, h/4, w/4

                # points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
                points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]),
                                                    unbiased=True, use_udp=True)
                points = np.concatenate([points[:, :, ::-1], prob], axis=2)


            # print epoch loss
            print(f'Epoch loss: {sum(epoch_loss) / len(epoch_loss)}')

            scheduler.step()
            

    # fp16 setting
    # ToDo: we should implement fp16 training
    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        raise NotImplementedError()
    
    # validation
    if validate:
        raise NotImplementedError()
