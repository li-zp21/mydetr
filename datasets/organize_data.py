import shutil
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json


def copy_images(image_dir, data_dir):
    filenames = sorted(os.listdir(image_dir))

    pbar = tqdm(filenames)
    for filename in pbar:
        pbar.set_description("Copying %s" % filename)
        image_p = os.path.join(image_dir, filename)
        if filename <= '2012_001051.jpg':
            target_dir = os.path.join(data_dir, 'images', 'train')
        else:
            target_dir = os.path.join(data_dir, 'images', 'val')

        os.makedirs(target_dir, exist_ok=True)
        target_p = os.path.join(target_dir, filename)
        shutil.copy2(image_p, target_p)


def build_annos(anno_dir, data_dir):
    filenames = sorted(os.listdir(anno_dir))
    names = {}
    category_id = 0
    image_id = 0
    train_anno_id = 0
    val_anno_id = 0

    images_train = []
    annotations_train = []
    images_val = []
    annotations_val = []
    categories = []

    pbar = tqdm(filenames)
    for filename in pbar:
        pbar.set_description("Copying %s" % filename)
        xml_p = os.path.join(anno_dir, filename)
        tree = ET.parse(xml_p)
        root = tree.getroot()
        
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in names.values():
                names[category_id] = name
                categories.append({
                    "id": int(category_id),
                    "name": name,
                })
                category_id += 1

        if filename <= '2012_001051.xml':
            images_train.append({
                "id": image_id,
                "width": int(root.find("size/width").text),
                "height": int(root.find("size/height").text),
                "file_name": filename.replace('.xml', '.jpg'),
            })
            for obj in root.findall("object"):
                name = obj.find("name").text
                xmin = int(obj.find("bndbox/xmin").text)
                ymin = int(obj.find("bndbox/ymin").text)
                xmax = int(obj.find("bndbox/xmax").text)
                ymax = int(obj.find("bndbox/ymax").text)
                w = xmax - xmin
                h = ymax - ymin
                annotations_train.append({
                    "id": train_anno_id,
                    "image_id": image_id,
                    "category_id": int(next(key for key, value in names.items() if value == name)),
                    "area": w * h,
                    "bbox": [xmin, ymin, w, h],
                    "iscrowd": 0,
                })
                train_anno_id += 1
            image_id += 1
        else:
            images_val.append({
                "id": image_id,
                "width": int(root.find("size/width").text),
                "height": int(root.find("size/height").text),
                "file_name": filename.replace('.xml', '.jpg'),
            })
            for obj in root.findall("object"):
                name = obj.find("name").text
                xmin = int(obj.find("bndbox/xmin").text)
                ymin = int(obj.find("bndbox/ymin").text)
                xmax = int(obj.find("bndbox/xmax").text)
                ymax = int(obj.find("bndbox/ymax").text)
                w = xmax - xmin
                h = ymax - ymin
                annotations_val.append({
                    "id": val_anno_id,
                    "image_id": image_id,
                    "category_id": int(next(key for key, value in names.items() if value == name)),
                    "area": w * h,
                    "bbox": [xmin, ymin, w, h],
                    "iscrowd": 0,
                })
                val_anno_id += 1
            image_id += 1

    target_dir = os.path.join(data_dir, 'annotations')
    os.makedirs(target_dir, exist_ok=True)
    train_anno_p = os.path.join(target_dir, 'train_annos.json')
    val_anno_p = os.path.join(target_dir, 'val_annos.json')

    train_annos = {
        "images": images_train,
        "annotations": annotations_train,
        "categories": categories,
    }
    val_annos ={
        "images": images_val,
        "annotations": annotations_val,
        "categories": categories,
    }
    with open(train_anno_p, 'w') as file:
        json.dump(train_annos, file, indent=4)
    with open(val_anno_p, 'w') as file:
        json.dump(val_annos, file, indent=4)
   
    # print(names.values())


def construct_dataset():
    image_dir = '../JPEGImages'
    anno_dir = '../Annotations'
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    copy_images(image_dir, data_dir)
    build_annos(anno_dir, data_dir)


if __name__ == "__main__":
    construct_dataset()
