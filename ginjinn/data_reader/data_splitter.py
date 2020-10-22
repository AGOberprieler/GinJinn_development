"""
Functions to generate a class-balanced training/validation/test split.
"""

import glob
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from typing import List, Union, Generator
import numpy as np
import pandas as pd


def split_dataset_dicts(
    dict_list: List[dict],
    class_names: List[str],
    task: str,
    p_val: Union[int, float] = 0,
    p_test: Union[int, float] = 0
    ) -> dict:
    """Create a new train/val/test split for a list of Detectron2 dataset dictionaries.
    This function requires user interaction.

    Parameters
    ----------
    dict_list : list of dict
        Image annotations in Detectron2's default format
    class_names : list of str
        Ordered list of object class names
    task : str
        "bbox-detection" or "instance-segmentation"
    p_val : int or float
        Proportion of images to be used as validation set
    p_test : int or float
        Proportion of images to be used as test set

    Returns
    -------
    dicts_split : dict
        For each subset ("train", "val", "test"), dicts_split[subset] contains
        a list of dictionaries, which can be registered as dataset for
        Detectron2.
    """
    class_counts = count_class_occurrences(dict_list, len(class_names), task)

    n_trials = 0
    accept = False
    while not accept:
        partition = greedy_split(class_counts, p_val, p_test)
        n_trials += 1

        col0 = class_counts[partition["train"], :].sum(axis=0)
        col1 = class_counts[partition["val"], :].sum(axis=0)
        col2 = class_counts[partition["test"], :].sum(axis=0)

        df1 = pd.DataFrame(
            [[len(partition["train"]), len(partition["val"]), len(partition["test"])]],
            columns=["train", "val", "test"],
            index=["images"]
        )
        df2 = pd.DataFrame(
            {"train": col0, "val": col1, "test": col2},
            columns=["train", "val", "test"],
            index=class_names
        )
        df = pd.concat([df1, df2])

        if p_val == 0:
            del df['val']
        if p_test == 0:
            del df['test']

        print(df)

        # handle invalid splits
        if 0 in df.values:
            if n_trials < 10:
                continue
            if confirmation("Could not find a valid split, try again?"):
                n_trials = 0
                continue
            sys.exit()

        accept = confirmation(
            "Do you want to accept this split? (Otherwise a new one will be generated.)"
        )

    dicts_split = dict()
    for key in partition:
        dicts_split[key] = [d for i, d in enumerate(dict_list) if i in partition[key]]

    return dicts_split


def split_dataset(
    ann_path: str,
    project_dir: str,
    dict_list: List[dict],
    class_names: List[str],
    task: str,
    ann_type: str,
    p_val: Union[int, float] = 0,
    p_test: Union[int, float] = 0
    ) -> dict:
    """Create a new train/val/test split for a given dataset.

    This function splits images along with annotations into subsets for training,
    validation and test. Splitted datasets are returned in Detectron2's
    default format and, additionally, stored within the project directory.

    Parameters
    ----------
    ann_path : str
        Annotations to be splitted, either a COCO json file or a directory
        containing PascalVOC xml files
    project_dir : str
        Project directory
    dict_list : list of dict
        Image annotations in Detectron2's default format
    class_names : list of str
        Ordered list of object class names
    task : str
        "bbox-detection" or "instance-segmentation"
    ann_type : str
        "COCO" or "PVOC"
    p_val : int or float
        Proportion of images to be used as validation set
    p_test : int or float
        Proportion of images to be used as test set

    Returns
    -------
    dicts_split : dict
        For each subset ("train", "val", "test"), dicts_split[subset] contains
        a list of dictionaries, which can be registered as dataset for
        Detectron2.
    """

    split_dir = os.path.join(project_dir, "data_split")
    if os.path.exists(split_dir):
        if confirmation(
            split_dir + ' already exists. Do you want do overwrite it? ' \
            '(If you want to reuse it, please type "no" and adjust the image ' \
            'and annotation paths in your config file accordingly.)'
        ):
            shutil.rmtree(split_dir)
        else:
            sys.exit()

    os.makedirs(os.path.join(split_dir, "annotations"))
    
    os.makedirs(os.path.join(split_dir, "images", "train"))
    if p_val > 0:
        os.mkdir(os.path.join(split_dir, "images", "val"))
    if p_test > 0:
        os.mkdir(os.path.join(split_dir, "images", "test"))

    if ann_type == "PVOC":
        os.makedirs(os.path.join(split_dir, "annotations", "train"))
        if p_val > 0:
            os.mkdir(os.path.join(split_dir, "annotations", "val"))
        if p_test > 0:
            os.mkdir(os.path.join(split_dir, "annotations", "test"))

    dicts_split = split_dataset_dicts(dict_list, class_names, task, p_val, p_test)

    # save splitted data
    if ann_type == "COCO":
        save_split_coco(ann_path, dicts_split, split_dir)
    elif ann_type == "PVOC":
        save_split_pvoc(ann_path, dicts_split, split_dir)

    return dicts_split


def count_class_occurrences(
    dict_list: List[dict],
    n_classes: int,
    task: str = None
    ) -> "np.ndarray[np.int]":
    """Count class occurences for each image of a Detectron2 dataset.

    Parameters
    ----------
    dict_list : list of dict
        Image annotations in Detectron2's default format
    n_classes : int
        Number of object classes
    task : str
        Unless task=None, objects are only considered if their annotation
        contains all information necessary for a specific task (either
        "bbox-detection" or "instance-segmentation").

    Returns
    -------
    class_counts: ndarray
        2-D array indicating how many objects of each class (column) are
        annotated within each image (row)
    """
    class_counts = np.zeros((len(dict_list), n_classes), dtype=int)

    if task == "bbox-detection":
        required = ("bbox", "bbox_mode", "category_id")
    elif task == "instance-segmentation":
        required = ("bbox", "bbox_mode", "category_id", "segmentation")
    elif task is None:
        required = ()

    for i_img, img_annotation in enumerate(dict_list):
        for obj_annotation in img_annotation["annotations"]:
            if None not in [obj_annotation.get(key) for key in required]:
                class_counts[i_img, obj_annotation["category_id"]] += 1

    return class_counts


def sel_order(
    n: int,
    p_val: Union[int, float] = 0,
    p_test: Union[int, float] = 0
    ) -> Generator[int, None, None]:
    """Order in which different data sets select their images.
    To avoid possible bias, this function makes sure that the data sets select
    their images at regular intervals from the beginning to the end of the
    splitting process.

    Parameters
    ----------
    n : int
        Total number of images
    p_val : int or float
        Proportion of images to be used as validation set
    p_test : int or float
        Proportion of images to be used as test set

    Yields
    -------
    j: int
        Index of next dataset (0=train, 1=validation, 2=test)
    """
    p = [1-p_val-p_test, p_val, p_test]

    if not (0 <= p_val <= 1 and 0 <= p_test <= 1):
        raise ValueError("Both validation and test proportion must be between 0.0 and 1.0.")

    counter = [0] * 3

    for i_img in range(n):
        j = np.argmax([p[i] - counter[i]/(i_img+1) for i in range(3)])
        yield j
        counter[j] = counter[j]+1


def greedy_split(
    class_counts: "np.ndarray[np.int]",
    p_val: Union[int, float] = 0,
    p_test: Union[int, float] = 0
    ) -> dict:
    """Randomized greedy test split algorithm.

    Parameters
    ----------
    class_counts : ndarray
        2-D array indicating how many objects of each class (column) are
        annotated within each image (row)
    p_val : int or float
        Proportion of images to be used as validation set
    p_test : int or float
        Proportion of images to be used as test set

    Returns
    -------
    partition: dict
        For each data subset, partition[subset] lists the corresponding image indices,
        i.e. row indices in class_counts,
        e.g. {'train': [2, 7, 9, 3, 8, 0], 'val': [5, 6], 'test': [4, 1]}
    """
    n = class_counts.shape[0]
    rest = list(range(n))
    sets = ["train", "val", "test"]
    partition = {"train": [], "val": [], "test": []}
    class_counts_dataset = {"train": 0, "val": 0, "test": 0}
    avg = np.mean(class_counts, axis=0)

    for i_set in sel_order(n, p_val, p_test):
        subset = sets[i_set]
        best = float("inf")
        i_best = rest[0]
        k = min(len(rest), 100, n//2)

        # select image
        for i in random.sample(range(len(rest)), k):
            i_row = rest[i]
            new = class_counts_dataset[subset] + class_counts[i_row, :]
            cost = np.square(new - avg * (1 + len(partition[subset]))).sum()
            if cost < best:
                best = cost
                i_best = i
        partition[subset].append(rest.pop(i_best))
        class_counts_dataset[subset] += class_counts[partition[subset][-1], :]

    return partition


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


def save_split_coco(
    ann_file: str,
    dicts_split: dict,
    split_dir: str
    ):
    """Save train/val/test split of a COCO dataset to disk.

    This function partitions a dataset with annotations in COCO format into
    subsets and stores them within a given directory. New image directories
    and new COCO json files are created. To avoid wasting disk space, images
    are not copied, but hard-linked into the new directories.

    Parameters
    ----------
    ann_file : str
        Path of COCO json file to be splitted
    dicts_split: dict
        For each dataset ("train", "val", "test"), partition[subset] is a
        list of image annotations in Detectron2's default dictionary format.
    split_dir : str
        Directory for storing newly created datasets
    """
    with open(ann_file, 'r') as json_file:
        ann_dict = json.load(json_file)

    info = ann_dict['info']
    licenses = ann_dict['licenses']
    images = ann_dict['images']
    annotations = ann_dict['annotations']
    categories = ann_dict['categories']

    for key in dicts_split:
        if not dicts_split[key]:
            continue

        img_dir = os.path.split(dicts_split[key][0]["file_name"])[0]
        img_ids = [record["image_id"] for record in dicts_split[key]]

        annotations_part = [ann for ann in annotations if ann["id"] in img_ids]
        images_part = [img for img in images if img["id"] in img_ids]

        json_new = os.path.join(split_dir, "annotations", key + ".json")
        with open(json_new, 'w') as json_file:
            json.dump({
                'info': info,
                'licenses': licenses,
                'images': images_part,
                'annotations': annotations_part,
                'categories': categories
                },
                json_file,
                indent=2,       # None/0 for more compact representation
                sort_keys=True
            )

        img_names_part = [os.path.split(img["file_name"])[1] for img in images_part]

        for fname in img_names_part:
            img_path_orig = os.path.join(img_dir, fname)
            img_path_new = os.path.join(split_dir, "images", key, fname)
            os.link(img_path_orig, img_path_new)


def save_split_pvoc(
    ann_dir: str,
    dicts_split: dict,
    split_dir: str
    ):
    """Save train/val/test split of a PascalVOC dataset to disk.

    This function partitions a dataset with annotations in PascalVOC format
    into subsets and stores them within a given directory. New directories for
    images and annotation xml files are created. To avoid wasting disk space,
    images and annotation files are not copied, but hard-linked into the new
    directories.

    Parameters
    ----------
    ann_file : str
        Path of COCO json file to be splitted
    dicts_split: dict
        For each dataset ("train", "val", "test"), partition[subset] is a
        list of image annotations in Detectron2's default dictionary format.
    split_dir : str
        Directory for storing newly created datasets
    """
    ann_map = dict()

    for ann_path in glob.glob(os.path.join(ann_dir, "*.xml")):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        img_name = os.path.split(root.findtext("filename"))[1]
        ann_map[img_name] = ann_path

    for key in dicts_split:
        for record in dicts_split[key]:
            img_path = record["file_name"]
            img_name = os.path.split(img_path)[1]
            img_path_new = os.path.join(split_dir, "images", key, img_name)
            os.link(img_path, img_path_new)

            ann_path = ann_map[img_name]
            ann_name = os.path.split(ann_map[img_name])[1]
            ann_path_new = os.path.join(split_dir, "annotations", key, ann_name)
            os.link(ann_path, ann_path_new)
