'''
Convert input to Detectron2's dictionary format.
'''

import glob
import os
from typing import List
import xml.etree.ElementTree as ET
from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode


def get_dicts(
        ann_path: str,
        img_path: str,
        ann_format: str,
        class_names: List[str]=None
    ) -> List[dict]:
    '''Generic function to load annotations into Detectron2 format.

    Parameters
    ----------
    ann_path : str
        COCO: path of json file
        Pascal VOC: directory containing xml files
    img_path : str
        Directory containing JPG images
    ann_format : str
        either "PVOC" or "COCO"
    class_names : list of str
        only required for Pascal VOC

    Returns
    -------
    dict_list: list of dict
        Annotations in Detectron2 format
    '''

    if ann_format == "COCO":
        dict_list = load_coco_json(ann_path, img_path)
    elif ann_format == "PVOC":
        dict_list = get_dicts_pvoc(ann_path, img_path, class_names)

    return dict_list


def get_class_names_pvoc(ann_dirs: List[str]) -> List[str]:
    '''Get all object class names contained in a number of Pascal VOC annotation files.

    Parameters
    ----------
    ann_dirs : list of str
        Directories containing annotation files, each xml file is scanned.

    Returns
    -------
    class_names: list of str
        Ordered list of object class names
    '''

    class_names = set()

    for ann_dir in ann_dirs:
        for ann_file in glob.glob(os.path.join(ann_dir, "*.xml")):
            tree = ET.parse(ann_file)
            root = tree.getroot()

            for obj in root.findall("object"):
                name = obj.findtext("name")
                class_names.add(name)

    class_names = list(class_names)
    class_names.sort()

    return class_names


# unused
def get_img_ids(img_dirs: List[str]) -> dict:
    '''Assign unique ID to each JPG image from multiple directories.

    Parameters
    ----------
    img_dirs : list of str
        Directories containing JPG images.

    Returns
    -------
    img_ids: dict
        key = image path, value = image ID
    '''

    i_img = 1
    img_ids = {}

    for img_dir in img_dirs:
        for img_file in os.scandir(img_dir):
            if os.path.splitext(img_file.path)[1] in ["jpg", "jpeg", "JPG", "JPEG"]:
                img_ids[img_file.path] = i_img
                i_img += 1

    return img_ids


def get_dicts_pvoc(ann_dir: str, img_dir: str, class_names: List[str]) -> List[dict]:
    '''Load Pascal VOC annotations to Detectron2 format.

    Parameters
    ----------
    ann_dir : str
        Directory containing xml files
    img_dir : str
        Directory containing JPG images
    class_names : list of str
        required to assign category IDs

    Returns
    -------
    dict_list: list of dict
        Annotations in Detectron2 format
    '''

    dict_list = []

    for i_img, ann_file in enumerate(glob.glob(os.path.join(ann_dir, "*.xml"))):

        tree = ET.parse(ann_file)
        root = tree.getroot()

        record = {}

        # folder = root.findtext("folder")
        filename = root.findtext("filename")
        record["file_name"] = os.path.join(img_dir, filename)
        record["image_id"] = i_img + 1

        size = root.find("size")
        record["height"] = int(size.findtext("height"))
        record["width"] = int(size.findtext("width"))

        record["annotations"] = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.findtext("xmin")) - 1
            ymin = int(bbox.findtext("ymin")) - 1
            xmax = int(bbox.findtext("xmax"))
            ymax = int(bbox.findtext("ymax"))
            name = obj.findtext("name")

            annotation = {
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': class_names.index(name)
            }

            record["annotations"].append(annotation)

        dict_list.append(record)

    return dict_list
