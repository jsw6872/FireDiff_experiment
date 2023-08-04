from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
import transforms as T
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.transforms as transforms
from PIL import Image, ImageDraw

import numpy as np
import json
import cv2
import os

import xml.etree.ElementTree as ET


def get_transform():
    transforms = A.Compose([
                            # A.Normalize(mean=mean2, std=std2, max_pixel_value=255),
                            A.HorizontalFlip(p=1),
                            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                            ToTensorV2(),
                        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    return transforms


def draw_box(pil, box, color = 2,width=2):
    draw = ImageDraw.Draw(pil)
    color_dict = {0 : 'blue', 1: 'green', 2 : 'red'}
    draw.rectangle(box, width=width, outline=color_dict[color], fill=None)
    return pil


def check_no_label_img(img_path, file_name_list, save_txt_path):
    # img_list = os.listdir(img_path)
    save_txt_path = f"{save_txt_path}/no_label_img.txt"
    for file_name in file_name_list:
        with open(save_txt_path, 'a' if os.path.exists(save_txt_path) else 'w') as file:
            # if file_name not in img_list:
            file.write(file_name + "\n")
                    
    


def save_bbox_jpg(xml_file, save_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    no_label_file_name_list = list()
    for images_info in tqdm(root.findall('image')):
        file_name = images_info.get('name')
        
        
        image = cv2.imread(f'./data/train/{file_name}')
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        
        points = list()

        if images_info.find('box') is None:
            no_label_file_name_list.append(f'{file_name.split("/")[-1][:-4]}.jpg')
        else:
            for box_info in images_info.findall('box'):
                x_min = float(box_info.get('xtl'))
                y_min = float(box_info.get('ytl'))
                x_max = float(box_info.get('xbr'))
                y_max = float(box_info.get('ybr'))
                
                point = (x_min, y_min, x_max, y_max)
                points.append(point)
            
                new_pil_image = draw_box(pil_image, point)
            
            numpy_image = np.array(new_pil_image)
            bbox_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            
            # cv2.imwrite(f'{save_path}/{file_name.split("/")[-1][:-4]}_bbox.jpg', bbox_img)
        
    return no_label_file_name_list


if __name__ == '__main__':
    dataset_path = './data' # Dataset 경로 지정 필요
    xml_file = dataset_path + '/' + 'controlnet_xml_2.xml'
    val_img_path = dataset_path
    
    save_path = './bbox_img/controlnet'
    
    file_name_list = save_bbox_jpg(xml_file, save_path)
    check_no_label_img('./data/train/controlnet-chatgpt', file_name_list, save_path)