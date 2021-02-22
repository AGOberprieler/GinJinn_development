"""
Module for generic helper functions.
"""

import sys
import json
import glob
import os
import xml
import xml.etree.ElementTree as et
from typing import List, Sequence, Tuple
import imantics
import numpy as np
from pycocotools import mask as pmask

def coco_seg_to_mask(seg, width, height):
    """coco_seg_to_mask

    Convert segmentation annotation (either list of polygons or COCO's compressed RLE)
    to binary mask.

    Parameters
    ----------
    seg : dict or list
        Segmentation annotation
    width : int
        Image/mask width
    height : int
        Image/mask height

    Returns
    -------
    seg_mask : np.ndarray
        Boolean segmentation mask

    Raises
    ------
    TypeError
        Raised for unsupported annotation formats.
    """
    if isinstance(seg, dict):
        # compressed rle to mask
        seg_mask = pmask.decode(seg).astype("bool")
    elif isinstance(seg, list):
        # polygon to mask
        polygons = imantics.Polygons(seg)
        seg_mask = polygons.mask(width, height).array
    else:
        raise TypeError(
            "Unknown segmentation format, polygons or compressed RLE expected"
        )
    return seg_mask

def bbox_from_mask(mask: np.ndarray, fmt: str):
    """Calculate bounding box from segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask
    fmt : str
        Output format, either "xywh" (COCO-like) or "xyxy" (PascalVoc-like)

    Returns
    -------
    np.ndarray
        Bounding box

    Raises
    ------
    ValueError
        Raised for unsupported output formats.
    """
    x_any = mask.any(axis=0)
    y_any = mask.any(axis=1)
    x = np.where(x_any)[0].tolist()
    y = np.where(y_any)[0].tolist()
    x1, y1, x2, y2 = (x[0], y[0], x[-1] + 1, y[-1] + 1)
    if fmt == "xywh":
        bbox = ( x1, y1, x2 - x1, y2 - y1 )
    elif fmt == "xyxy":
        bbox = ( x1, y1, x2, y2 )
    else:
        raise ValueError(
            f"Unknown bounding box format \"{fmt}\"."
        )
    return np.array(bbox)

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

def plot_pvoc_annotated_img(
    img: np.ndarray,
    ann: xml.etree.ElementTree.ElementTree,
    ax=None,
):
    '''plot_pvoc_annotated_img

    Plot PVOC annotated image.

    Parameters
    ----------
    img : np.ndarray
        Image as numpy array
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    ax
        matplotlib axis, by default None

    Returns
    -------
    matplotlib axis
    '''
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(img)
    overlay_pvoc_ann(ann, ax)

    return ax

def overlay_pvoc_ann(
    ann: xml.etree.ElementTree.ElementTree,
    ax,
):
    '''overlay_pvoc_ann

    Plot PVOC annotation on ax.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    ax
        matplotlib axis

    Returns
    -------
    matplotlib axis
    '''
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for obj in get_pvoc_objects(ann):
        bbox = get_pvoc_obj_bbox(obj)
        w, h = bbox_size(bbox)
        rect = Rectangle(
            [bbox[0], bbox[1]], w, h,
            facecolor=None,
            fill=False,
            edgecolor='orange'
        )
        ax.add_patch(rect)

    return ax

def load_pvoc_annotation(
    ann_path: str,
) -> xml.etree.ElementTree.ElementTree:
    '''load_pvoc_annotation

    Load PVOC annotations from file.

    Parameters
    ----------
    ann_path : str
        PVOC annotation XML file path

    Returns
    -------
    xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    '''
    return et.parse(ann_path)

def write_pvoc_annotation(
    ann: xml.etree.ElementTree.ElementTree,
    ann_file: str,
):
    '''write_pvoc_annotation

    Write PVOC annotation in ElementTree format to XML file.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    ann_file : str
        Path to annotation XML file
    '''
    import xml.dom.minidom as minidom

    root = ann.getroot()
    xmlstr = minidom.parseString(et.tostring(root)).toprettyxml(indent='  ')
    with open(ann_file, 'w') as ann_f:
        ann_f.write(xmlstr)

def clip_bbox(
    bbox: Sequence[float],
    clipping_range: Sequence[float],
) -> Sequence[float]:
    '''clip_bbox

    Clip bounding box.

    Parameters
    ----------
    bbox : Sequence[float]
        Bounding-box in x0y0x1y1 format.
    clipping_range : Sequence[float]
        Clipping range in x0x1y0y1 format.

    Returns
    -------
    Sequence[float]
        Clipped bounding box in x0y0x1y1 format.
    '''
    xmn, xmx, ymn, ymx = clipping_range
    bbox_clipped = np.clip(
        bbox,
        [xmn, ymn, xmn, ymn],
        [xmx, ymx, xmx, ymx],
    )
    return bbox_clipped

def crop_bbox(
    bbox: Sequence[float],
    cropping_range: Sequence[float],
) -> Sequence[float]:
    '''crop_bbox

    Crop bounding box. Clips bbox and converts coordinates
    to local coordinates in cropping range.

    Parameters
    ----------
    bbox : Sequence[float]
        Bounding-box in x0y0x1y1 format.
    cropping_range : Sequence[float]
        Cropping range in x0x1y0y1 format.

    Returns
    -------
    Sequence[float]
        Cropped bounding box in x0y0x1y1 format.
    '''

    bbox_clipped = clip_bbox(bbox, cropping_range)
    bbox_cropped = [
        bbox_clipped[0] - cropping_range[0],
        bbox_clipped[1] - cropping_range[2],
        bbox_clipped[2] - cropping_range[0],
        bbox_clipped[3] - cropping_range[2],
    ]

    return bbox_cropped

def bbox_size(
    bbox: Sequence[float],
) -> Tuple[float, float]:
    '''bbox_size

    Calculate bounding box size (width, height).

    Parameters
    ----------
    bbox : Sequence[float]
        Bounding-box in x0y0x1y1 format.

    Returns
    -------
    Tuple[float, float]
        Tuple of (width, height)
    '''
    return (
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    )

def bbox_area(
    bbox: Sequence[float],
) -> float:
    '''bbox_area

    Calculate bounding-box area.

    Parameters
    ----------
    bbox : Sequence[float]
        Bounding-box in x0y0x1y1 format.

    Returns
    -------
    float
        Area of the bounding-box.
    '''
    w, h = bbox_size(bbox)
    return w * h

def get_pvoc_filename(
    ann: xml.etree.ElementTree.ElementTree,
) -> str:
    '''get_pvoc_filename

    Get image file name from PVOC annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree

    Returns
    -------
    str
        Image file name.
    '''
    return ann.find('filename').text

def set_pvoc_filename(
    ann: xml.etree.ElementTree.ElementTree,
    filename: str,
):
    '''set_pvoc_filename

    Set image file name for PVCO annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    filename : str
        Image file name
    '''
    ann.find('filename').text = filename

def get_pvoc_size(
    ann: xml.etree.ElementTree.ElementTree,
) -> Sequence[int]:
    '''get_pvoc_size

    Get size of annotated image from PVOC annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree

    Returns
    -------
    Sequence[int]
        Image size as Tuple (width, height, depth)
    '''
    size_node = ann.find('size')
    return [
        int(size_node.find('width').text),
        int(size_node.find('height').text),
        int(size_node.find('depth').text),
    ]

def set_pvoc_size(
    ann: xml.etree.ElementTree.ElementTree,
    size: Sequence[int],
):
    '''set_pvoc_size

    Set size value for PVOC annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    size : Sequence[int]
        Size as sequence of width, height, depth.
    '''
    size_node = ann.find('size')
    size_node.find('width').text = str(size[0])
    size_node.find('height').text = str(size[1])
    size_node.find('depth').text = str(size[2])

def get_pvoc_objects(
    ann: xml.etree.ElementTree.ElementTree,
) -> List[xml.etree.ElementTree.ElementTree]:
    '''get_pvoc_objects

    Get a list of PVCO annotation objects in ElementTree format.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree

    Returns
    -------
    List[xml.etree.ElementTree.ElementTree]
        List of PVOC objects as ElementTree
    '''
    return ann.findall('object')

def add_pvoc_object(
    ann: xml.etree.ElementTree.ElementTree,
    obj: xml.etree.ElementTree.ElementTree,
):
    '''add_pvoc_object

    Add PVOC object to PVOC annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    obj : xml.etree.ElementTree.ElementTree
        PVOC object as ElementTree
    '''
    r = ann.getroot()
    r.append(obj)

def drop_pvoc_objects(
    ann: xml.etree.ElementTree.ElementTree,
):
    '''drop_pvoc_objects

    Remove all objects from PVOC annotation.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    '''
    r = ann.getroot()
    for o in get_pvoc_objects(ann):
        r.remove(o)

def get_pvoc_obj_bbox(
    obj: xml.etree.ElementTree.ElementTree,
) -> Sequence[int]:
    '''get_pvoc_obj_bbox

    Get bounding-box from PVOC object.

    Parameters
    ----------
    obj : xml.etree.ElementTree.ElementTree
        PVOC object as ElementTree

    Returns
    -------
    Sequence[int]
        Bounding-box in x0y0x1y1 format.
    '''
    bbox_node = obj.find('bndbox')
    bbox = [
        int(bbox_node.find('xmin').text),
        int(bbox_node.find('ymin').text),
        int(bbox_node.find('xmax').text),
        int(bbox_node.find('ymax').text),
    ]
    return bbox

def set_pvoc_obj_bbox(
    obj: xml.etree.ElementTree.ElementTree,
    bbox: Sequence[int],
):
    '''set_pvoc_obj_bbox

    Set bounding-box for PVOC object.

    Parameters
    ----------
    obj : xml.etree.ElementTree.ElementTree
        PVOC object as ElementTree
    bbox : Sequence[int]
        Bounding-box in x0y0x1y1 format.
    '''
    bbox_node = obj.find('bndbox')
    bbox_node.find('xmin').text = str(bbox[0])
    bbox_node.find('ymin').text = str(bbox[1])
    bbox_node.find('xmax').text = str(bbox[2])
    bbox_node.find('ymax').text = str(bbox[3])
