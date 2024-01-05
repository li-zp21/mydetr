"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import os

from pathlib import Path, PurePath



def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    color_palette = sns.color_palette(n_colors=len(logs))
    for df, color in zip(dfs, color_palette):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                _df = df.select_dtypes(include=['number'])
                _df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
        
    lines = [mlines.Line2D([], [], color=color, label=log) for color, log in zip(color_palette, logs)]
    for ax, field in zip(axs, fields):
        ax.legend(handles=lines)
        ax.set_title(field)
    plt.savefig('logs.png')


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette(n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend([f.parts[0] for f in files])
    axs[1].set_title('Scores / Recall')
    axs[1].legend([f.parts[0] for f in files])
    plt.savefig('precision_recall.png')
    return fig, axs


# Class labels 
class_labels = [ 
    "bottle", "tvmonitor", "train", "person", "sofa", "pottedplant", "chair",
    "motorbike", "boat", "dog", "bird", "bicycle", "diningtable", "cat",
    "horse", "bus", "car", "sheep", "aeroplane", "cow"
]


def plot_imgs(targets, results, image_dir, output_dir):
    # Getting the color map from matplotlib 
    colour_map = plt.get_cmap("tab20b") 
    # Getting 20 different colors from the color map for 20 different classes 
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))] 
    
    for target, result in zip(targets, results):
        target = {k: v.cpu().numpy() for k, v in target.items()}
        result = {k: v.cpu().numpy() for k, v in result.items()}
        image_id = target["image_id"].item()
        
        if image_id >= 1541:
            img_dir = os.path.join(image_dir, "val")
            image_id -= 1541
        else:
            img_dir = os.path.join(image_dir, "train")
        
        image_filename = sorted(os.listdir(img_dir))[image_id]
        image_path = os.path.join(img_dir, image_filename)
        image = Image.open(image_path)
        
        scores, labels, boxes = result["scores"], result["labels"], result["boxes"]
        scores, labels, boxes = scores[scores > 0.5], labels[scores > 0.5], boxes[scores > 0.5]
        keep = nms(boxes, scores, iou_threshold=0.5)
        scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
        
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        
        for score, label, box in zip(scores, labels, boxes):
            try:
                rect = patches.Rectangle(
                    (box[0], box[1]), 
                    box[2] - box[0], 
                    box[3] - box[1],
                    linewidth=2, 
                    edgecolor=colors[int(label)], 
                    facecolor="none",
                )
                ax.add_patch(rect)
                        
                plt.text( 
                    box[0], 
                    box[1], 
                    s=f'{class_labels[int(label)]}, {score:.2f}', 
                    color="white", 
                    verticalalignment="top", 
                    bbox={"color": colors[int(label)], "pad": 0}, 
                ) 
            except IndexError:
                pass
            
        save_dir = Path(output_dir, "images")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, image_filename)
        plt.savefig(save_path)
        plt.close()
        
        
def nms(boxes, scores, iou_threshold):
    """
    非极大值抑制。
    :param boxes: 边界框的数组，每个元素的格式为 [x_min, y_min, x_max, y_max]。
    :param scores: 每个边界框对应的置信度分数。
    :param iou_threshold: IOU阈值用于决定是否抑制。
    :return: 抑制后剩余的边界框的索引。
    """
    # boxes = boxes[scores >= threshold]
    # scores = scores[scores >= threshold]
    # 计算每个边界框的面积
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 按照置信度对边界框进行排序
    sorted_indices = np.argsort(scores)

    keep = []
    while sorted_indices.size > 0:
        # 取出得分最高的边界框
        current_box_index = sorted_indices[-1]
        current_box = boxes[current_box_index]
        keep.append(current_box_index)
        if sorted_indices.size == 1:
            break
        sorted_indices = sorted_indices[:-1]
        other_boxes = boxes[sorted_indices]

        # 计算IOU
        xx1 = np.maximum(current_box[0], other_boxes[:, 0])
        yy1 = np.maximum(current_box[1], other_boxes[:, 1])
        xx2 = np.minimum(current_box[2], other_boxes[:, 2])
        yy2 = np.minimum(current_box[3], other_boxes[:, 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        intersection = w * h
        iou = intersection / (area[current_box_index] + area[sorted_indices] - intersection)

        # 去除IOU大于阈值的边界框
        sorted_indices = sorted_indices[iou < iou_threshold]

    return keep