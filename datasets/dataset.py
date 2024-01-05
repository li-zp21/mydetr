from pathlib import Path
import os
import torch
import torch.utils.data
import torchvision

import datasets.transforms as T


class DetrDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(DetrDataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = prepare

    def __getitem__(self, idx):
        img, target = super(DetrDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {
            'image_id': image_id, 
            'annotations': target
            }
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    
    
def prepare(image, target):
    w, h = image.size

    image_id = target["image_id"]
    image_id = torch.tensor([image_id])

    anno = target["annotations"]

    anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    classes = [obj["category_id"] for obj in anno]
    classes = torch.tensor(classes, dtype=torch.int64)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes
    target["image_id"] = image_id

    # for conversion to coco api
    area = torch.tensor([obj["area"] for obj in anno])
    iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
    target["area"] = area[keep]
    target["iscrowd"] = iscrowd[keep]

    target["orig_size"] = torch.as_tensor([int(h), int(w)])
    target["size"] = torch.as_tensor([int(h), int(w)])

    return image, target


def make_transforms(image_set, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(args.img_norm_mean, args.img_norm_std)
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(args.flip_prob),
            T.RandomNoise(args.noise_factor),
            # T.Save(),
            normalize,
        ])
    else:
        return T.Compose([
            normalize,
        ])


def build_dataset(image_set, args):
    root = Path(args.data_dir)
    assert root.exists(), f'provided data path {root} does not exist'

    img_dirs = {
        "train": Path(root, 'images', 'train'),
        "val": Path(root, 'images', 'val'),
    }
    anno_files = {
        "train": Path(root, 'annotations', 'train_annos.json'),
        "val": Path(root, 'annotations', 'val_annos.json'),
    }
    img_dir = img_dirs[image_set]
    anno_file = anno_files[image_set]
    
    dataset = DetrDataset(img_dir, anno_file, transforms=make_transforms(image_set, args))
    return dataset