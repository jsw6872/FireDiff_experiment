import torch
import torchvision
from torchvision.io.image import read_image
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from dataset import COCO_dataformat, get_train_transform, get_valid_transform

import engine
from pycocotools.coco import COCO
import utils
import cv2
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
# import transforms as T


# def get_transform():
#     # transforms = []
#     # transforms.append(T.PILToTensor())
#     # transforms.append(T.ConvertImageDtype(torch.float))
    
#         # transforms.append(T.RandomHorizontalFlip(0.5))
#     mean1 = [90, 100, 100]
#     std1 = [30, 32, 28]
#     mean2 = [mean1[0]/255, mean1[1]/255, mean1[2]/255]
#     std2 = [std1[0]/255, std1[1]/255, std1[2]/255]
    
#     transforms = A.Compose([
#                             A.Normalize(mean=mean2, std=std2, max_pixel_value=255),
#                             A.HorizontalFlip(p=0.5),
#                             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#                             ToTensorV2(),
#                         ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
#     return transforms


if __name__ == '__main__':
    # dataset_path = '/home/work/jsw_workspace/detection/fire_data/' # Dataset 경로 지정 필요
    dataset_path = './data/'  # Dataset 경로 지정 필요
    train_path = dataset_path + 'train/train.json'
    val_path = dataset_path + 'test/test.json'

    train_img_path = dataset_path + 'train/fire'
    val_img_path = dataset_path + 'test/fire'
    
    coco = COCO(train_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    dataset = COCO_dataformat(train_img_path, train_path, get_train_transform())#, get_transform())
    dataset_test = COCO_dataformat(val_img_path, val_path, get_valid_transform())#, get_transform())

    indices = torch.randperm(len(dataset)).tolist()

    data_loader = torch.utils.data.DataLoader(
                                            dataset, batch_size=8, shuffle=True, num_workers=8,
                                            collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
                                                dataset_test, batch_size=4, shuffle=False, num_workers=4,
                                                collate_fn=utils.collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights = None, num_classes=2, weights_backbone = ResNet50_Weights.IMAGENET1K_V2, trainable_backbone_layers=3)# ResNet50_Weights.IMAGENET1K_V2)#, weights_backbone = None)
    # for child in model.children():
    #     for param in child.parameters():
    #         param.requires_grad = False
    # 분류기를 새로운 것으로 교체하는데, num_classes는 사용자가 정의합니다
    
    # 분류기에서 사용할 입력 특징의 차원 정보를 얻습니다
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # num_classes = len(coco.cats.keys())

    # 미리 학습된 모델의 머리 부분을 새로운 것으로 교체합니다
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    
    model.to(device)
    print('model is done')

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.05, momentum=0.9, weight_decay=0.0001)
    num_epochs = 20
    loss_list = []
    for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
        metric_logger, all_losses_dict_df = engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        epoch_loss = all_losses_dict_df.apply(lambda x: sum(x), axis=1).mean()
        loss_list.append(epoch_loss)
        
        # if epoch_loss <= min(loss_list):
        #     best_model_wts = copy.deepcopy(model.state_dict())
        # evaluate on the test dataset
        print(f'---------{epoch+1}epoch evaluation---------')
        evaluation = engine.evaluate(model, data_loader_test, device=device)

        best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        torch.save(model,f'./model/aug_{epoch+1}.pt')
    
    print("Done!")

