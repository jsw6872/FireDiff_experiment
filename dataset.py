import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import utils
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
        # images = images_copy.transpose(2,0,1).copy()
        # images /= 255.0 # albumentations 라이브러리로 toTensor 사용시 normalize 안해줘서 미리 해줘야~
        # images = torch.tensor(images)
        
        ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
        anns = self.coco.loadAnns(ann_ids)

        target = {}

        coco_bboxes = []
        bboxes = []
        labels = []
        area = []
        # image_id = []
        # masks = []
        for i in range(len(anns)):
            bbox = anns[i]['bbox']
            # masks.append(anns[i]["segmentation"])
            area.append(anns[i]['area'])
            coco_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            bboxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            labels.append(anns[i]['category_id'])
            # image_id.append(anns[i]['image_id'])

        boxes = torch.as_tensor(bboxes, dtype = torch.float32)
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype = torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor(index)
        # image_id = torch.as_tensor(image_id)
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        class_labels = ['fire' for _ in range(len(target['labels']))]
        if self.transforms is not None:
            transformed = self.transforms(image=images_copy, bboxes=coco_bboxes, class_labels = class_labels)
            target["boxes"] = transformed["bboxes"]
            images = transformed["image"]
            images = np.transpose(images,(0,1,2))
        else:
            images = images_copy.transpose(2,0,1).copy()
        images /= 255.0 # albumentations 라이브러리로 toTensor 사용시 normalize 안해줘서 미리 해줘야~
        images = torch.tensor(images)
            
        return images, target

            # if self.transform is not None:
            #     transformed = self.transform(image=images, mask=masks)
            #     images = transformed["image"]
            #     masks = transformed["mask"]
            # return images, masks, image_infos
        
        # if self.mode == 'test':
        #     if self.transform is not None:
        #         transformed = self.transform(image=images)
        #         images = transformed["image"]
        #     return images, image_infos

    def __len__(self):
        return len(self.coco.getImgIds()) # 전체 dataset의 size 반환 


def get_transform():
    # transforms = []
    # transforms.append(T.PILToTensor())
    # transforms.append(T.ConvertImageDtype(torch.float))
    
        # transforms.append(T.RandomHorizontalFlip(0.5))
    mean1 = [90, 100, 100]
    std1 = [30, 32, 28]
    mean2 = [mean1[0]/255, mean1[1]/255, mean1[2]/255]
    std2 = [std1[0]/255, std1[1]/255, std1[2]/255]
    
    transforms = A.Compose([
                            A.Normalize(mean=mean2, std=std2, max_pixel_value=255),
                            A.HorizontalFlip(p=0.5),
                            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                            ToTensorV2(),
                        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    return transforms


if __name__ == '__main__':
    dataset_path = '/home/work/jsw_workspace/detection/fire_data/train/' # Dataset 경로 지정 필요
    train_json_path = dataset_path + 'train.json'
    img_path = dataset_path + 'fire'
    dataset = COCO_dataformat(img_path, train_json_path, get_transform())

    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                                                    collate_fn=utils.collate_fn)

    batch = iter(train_data_loader)
    img, target = next(batch)
    print(True)