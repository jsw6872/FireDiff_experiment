import json
import os
from tqdm import tqdm

def converter(file_path):
    new_json = {}
    new_json['images'] = []
    new_json['categories'] = [
        {
            "id": 0,
            "name": "Fire",
            "supercategory": "Fire"
        }
    ]
    new_json['annotations'] = []

    file_path_list = os.listdir(file_path)
    file_path_list = list(map(lambda x: file_path+x,file_path_list)) 

    anno_id = 0
    for img_idx, file_name in tqdm(enumerate(file_path_list)):
        with open(file_name, 'r', encoding='utf-8-sig') as json_file:
            data = json.load(json_file)

            images_info = {}
            images_info['file_name'] =  data['image']['filename']
            images_info['height'] =  data['image']['resolution'][0]
            images_info['width'] =  data['image']['resolution'][1]
            images_info['id'] =  img_idx

            new_json['images'].append(images_info)

            for annotation in data['annotations']:
                annotations_info = {}
                if annotation['class'] == '04':
                    annotations_info['id'] = anno_id
                    annotations_info['image_id'] = img_idx
                    annotations_info['category_id'] = 0

                    annotations_info['bbox'] = [
                        annotation['box'][0],
                        annotation['box'][1],
                        annotation['box'][2]-annotation['box'][0],
                        annotation['box'][3]-annotation['box'][1]
                    ]
                    annotations_info['area'] = annotations_info['bbox'][2]*annotations_info['bbox'][3]
                    annotations_info['segmentation'] = []
                    annotations_info['iscrowd'] = 0

                    new_json['annotations'].append(annotations_info)
                    
                    anno_id += 1

    with open('./data/test/test.json', 'w') as file:
        json.dump(new_json, file, indent=4)



if __name__ == '__main__':
    data_path = './data/test/json/'
    converter(data_path)