import json
import os
from tqdm import tqdm

def converter(json_list):
    coco_dict = {}
    coco_dict['images'] = []
    coco_dict['categories'] = [
        {
            "id": 1,
            "name": "Fire",
            "supercategory": "Fire"
        }
    ]
    coco_dict['annotations'] = []


    image_id = 0
    anno_id = 1
    for json_file in tqdm(json_list):
        with open(json_file, 'r', encoding='utf-8-sig') as json_file:
            data = json.load(json_file)

            for image_info_dict in data['images']:
                origin_image_id = image_info_dict['id']

                for anno_info_dict in data['annotations']:
                    if anno_info_dict['image_id'] == origin_image_id:
                        anno_info_dict['id'] = anno_id
                        anno_info_dict['image_id'] = image_id

                        coco_dict['annotations'].append(anno_info_dict)
                    else:
                        continue

                    anno_id += 1

                image_info_dict['id'] = image_id
                coco_dict['images'].append(image_info_dict)

                image_id += 1

    with open('./data/train/total_train.json', 'w') as file:
        json.dump(coco_dict, file, indent=4)


if __name__ == '__main__':
    file_path = './data/train/'
    integrated_list = [f'{file_path}sd14-lora-fire-aihub-new-blip-best-1-annotations.json',
                    f'{file_path}sd14-lora-fire-aihub-new-chatgpt-annotations.json', 
                    f'{file_path}fire_train.json']
    
    converter(integrated_list)



