import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import utils
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import ops
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from torchvision import transforms as T
# import transforms as T


import albumentations as A
from albumentations.pytorch import ToTensorV2


class COCO_dataformat(Dataset):
    def __init__(self, img_path, data_dir, transforms=None):
        super().__init__()
        self.img_path = img_path
        # self.mode = mode
        self.transforms = transforms
        self.coco = COCO(data_dir)

        self.cat_ids = self.coco.getCatIds() # category id 반환
        self.cats = self.coco.loadCats(self.cat_ids) # category id를 입력으로 category name, super category 정보 담긴 dict 반환
        self.classNameList = ['Backgroud'] # class name 저장 
        for i in range(len(self.cat_ids)):
          self.classNameList.append(self.cats[i]['name'])

    def __getitem__(self, index):
        # image_id = self.coco.getImgIds(imgIds=index) # img id 또는 category id 를 받아서 img id 반환
        image_infos = self.coco.loadImgs(index)[0] # img id를 받아서 image info 반환
        
        # img_path = os.path.join(self.dataset_path, image_infos['file_name'])
        # images = Image.open(img_path).convert("RGB")

        # cv2 를 활용하여 image 불러오기(BGR -> RGB 변환 -> numpy array 변환 -> normalize(0~1))
        images = cv2.imread(os.path.join(self.img_path, image_infos['file_name']))
        images_copy = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)        
        # images_copy /= 255.0 # albumentations 라이브러리로 toTensor 사용시 normalize 안해줘서 미리 해줘야~
        
        ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
        anns = self.coco.loadAnns(ann_ids)

        target = {}

        coco_bboxes = []
        bboxes = []
        labels = []
        area = []
        # masks = []

        for i in range(len(anns)):
            bbox = anns[i]['bbox']
            # masks.append(anns[i]["segmentation"])
            area.append(anns[i]['area'])
            coco_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])

            bboxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])

            labels.append(anns[i]['category_id'])

        boxes = torch.as_tensor(bboxes, dtype = torch.float32)
        area = torch.as_tensor(area, dtype = torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor(index)
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        coco_boxes = torch.as_tensor(coco_bboxes, dtype= torch.float32)

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target['coco_boxes'] = coco_boxes
        # target["masks"] = masks
        
        class_labels = ['fire' for _ in range(len(target['labels']))]

        if self.transforms:
            transformed = self.transforms(image=images_copy, bboxes=target['coco_boxes'], labels=labels)
        
            if len(transformed['bboxes']) > 0:
                target["boxes"] = ops.box_convert(torch.as_tensor(transformed['bboxes'], dtype = torch.float32), 'xywh', 'xyxy')
            else:
                target["boxes"] = torch.zeros((0,4),dtype=torch.float32)
            images = transformed["image"]
            images = np.transpose(images, (0, 1, 2))
        
        images /= 255.0

        return image_infos['file_name'], images, target

    def __len__(self):
        return len(self.coco.getImgIds())  # 전체 dataset의 size 반환


def get_train_transform():
    mean1 = [90, 100, 100]
    std1 = [30, 32, 28]
    mean2 = [mean1[0]/255, mean1[1]/255, mean1[2]/255]
    std2 = [std1[0]/255, std1[1]/255, std1[2]/255]
    
    transforms = A.Compose([A.Resize(512,512),
                            A.HorizontalFlip(p=0.5),
                            A.ColorJitter(brightness=0.2, contrast=0, saturation=0.2, p=0.2),
                            A.GaussNoise(var_limit=(5, 15), p=0.2),
                            # A.Normalize(), #A.Normalize(mean=mean2, std=std2, max_pixel_value=255),
                            ToTensorV2(),
                        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    return transforms


def get_valid_transform():
    transforms = A.Compose([A.Resize(512,512),
                            ToTensorV2(),
                        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    return transforms


if __name__ == '__main__':
    dataset_path = '/home/yongchoooon/workspace/seongwoo/FireDiff_experiment/data/train/'  # Dataset 경로 지정 필요
    train_json_path = dataset_path + 'train.json'

    img_path = dataset_path + 'fire'
    dataset = COCO_dataformat(img_path, train_json_path, get_train_transform())

    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                                                    collate_fn=utils.collate_fn)

    batch = iter(train_data_loader)
    _, img, target = next(batch)

    print(target)
