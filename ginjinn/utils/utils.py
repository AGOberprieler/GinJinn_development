"""
Module for generic helper functions.
"""

import sys
import json
import glob
import os
from typing import List
import numpy as np

def confirmation(question: str) -> bool:
    """Ask question expecting "yes" or "no".

    Parameters
    ----------
    question : str
        Question to be printed

    Returns
    -------
    bool
        True or False for "yes" or "no", respectively
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}

    while True:
        choice = input(question + " [y/n]\n").strip().lower()
        if choice in valid.keys():
            return valid[choice]
        print("Please type 'yes' or 'no'\n")

def confirmation_cancel(question: str) -> bool:
    '''Ask question expecting 'yes' or 'no'.

    Parameters
    ----------
    question : str
        Question to be printed

    Returns
    -------
    bool
        True or False for 'yes' or 'no', respectively
    '''
    valid = {'yes': True, 'y': True, 'no': False, 'n': False}
    cancel = ['c', 'cancel', 'quit', 'q']

    while True:
        choice = input(question + ' [y(es)/n(o)/c(ancel)]\n').strip().lower()
        if choice in valid.keys():
            return valid[choice]
        elif choice in cancel:
            sys.exit()
        print('Please type "yes" or "no" (or "cancel")\n')

def get_image_files(
    img_dir: str,
    img_file_extensions: List[str] = [
        '*.jpg', '*.JPG', '*.jpeg', '*.JPEG',
        '*.png', '*.PNG',
    ],
) -> List[str]:
    '''get_image_files

    Get paths of image files in img_dir.

    Parameters
    ----------
    img_dir : str
        Directory containing images.
    img_file_extensions : List[str], optional
        Image file extensions,
        by default [ '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', ]

    Returns
    -------
    List[str]
        List of image file paths.
    '''
    img_files = []
    for ext in img_file_extensions:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    return img_files

def load_coco_ann(ann_path: str) -> dict:
    '''load_coco_ann

    Load coco JSON annotation into dict.

    Parameters
    ----------
    ann_path : str
        Path to annotation JSON file.

    Returns
    -------
    dict
        [COCO annotation as dict.
    '''
    with open(ann_path) as ann_f:
        ann = json.load(ann_f)
    return ann

def get_obj_anns(img_ann, ann: dict) -> List[dict]:
    '''get_obj_anns

    Get object annotations for image from COCO annotation dict.

    Parameters
    ----------
    img_ann
        Image id or image COCO dict.
    ann : dict
        COCO annotation ("dataset").

    Returns
    -------
    List[dict]
        List of COCO object annotations for img_ann.
    '''
    if isinstance(img_ann, int):
        img_id = img_ann
    else:
        img_id = img_ann['id']

    obj_anns = [obj_ann for obj_ann in ann['annotations'] if obj_ann['image_id'] == img_id]
    return obj_anns

def plot_coco_annotated_img(img, obj_anns: List[dict], ax=None):
    '''plot_coco_annotated_img

    Plot COCO annotations on image.

    Parameters
    ----------
    img
        Image, numpy array.
    obj_anns : List[dict]
        List of COCO object annotation dicts.
    ax
        pyplot axis to plot on, by default None.

    Returns
    -------
    pyplot axis
    '''

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(img)
    overlay_coco_annotations(obj_anns, ax)

    return ax

def overlay_coco_annotations(
    obj_anns: List[dict],
    ax
):
    '''overlay_coco_annotations

    Overlay coco annotations.

    Parameters
    ----------
    obj_anns : List[dict]
        List of COCO object annotation dicts.
    ax
        pyplot axis to plot on.

    Returns
    -------
    pyplot axis
    '''

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon

    for ann in obj_anns:
        print(ann)
        seg = ann.get('segmentation', None)
        if seg:
            poly = Polygon(
                np.array(seg).reshape(-1, 2),
                fill='orange',
                edgecolor='orange',
                alpha=0.5,
            )
            ax.add_patch(poly)

        bbox = ann.get('bbox', None)
        if bbox:
            rect = Rectangle(
                bbox[0:2], bbox[2], bbox[3],
                facecolor=None,
                fill=False,
                edgecolor='orange'
            )
            ax.add_patch(rect)

    return ax
