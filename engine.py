import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from dataset import COCO_dataformat, get_transform

import numpy as np
import pandas as pd
import copy

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch+1}]"
    all_losses = []
    all_losses_dict = []
    # lr_scheduler = None
    # if epoch == 0:
    #     warmup_factor = 1.0 / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)

        # lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        # )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses.item()

        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_value)
            sys.exit(1)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        
        # if lr_scheduler is not None:
        #     lr_scheduler.step()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
              
    all_losses_dict_df = pd.DataFrame(all_losses_dict)

    print(f"""
    Epoch {epoch+1},
    loss:{all_losses_dict_df.apply(lambda x: sum(x), axis=1).mean()},
    loss_classifier: {all_losses_dict_df["loss_classifier"].mean()},
    loss_box:{all_losses_dict_df["loss_box_reg"].mean()},
    loss_rpn_box: {all_losses_dict_df["loss_rpn_box_reg"].mean()},
    loss_object: {all_losses_dict_df["loss_objectness"].mean()} 
    """)
    return metric_logger, all_losses_dict_df


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    # n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)
    if device != 'cuda':
        device = torch.device("cpu")

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    # torch.set_num_threads(n_threads)
    return coco_evaluator



if __name__ == '__main__':
    dataset_path = '/home/work/jsw_workspace/detection/fire_data/train/' # Dataset 경로 지정 필요
    train_json_path = dataset_path + 'train.json'
    img_path = dataset_path + 'fire'
    dataset = COCO_dataformat(img_path, train_json_path)

    data_loader_test_coco = torch.utils.data.DataLoader(
                                            dataset, batch_size=1, shuffle=False, num_workers=0,
                                            collate_fn=utils.collate_fn)

    coco = get_coco_api_from_dataset(data_loader_test_coco.dataset)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    coco_eval = evaluate(model, data_loader_test_coco, device)
    print(True)
    # iou_types = _get_iou_types(model)
    # coco_evaluator = CocoEvaluator(coco, iou_types)