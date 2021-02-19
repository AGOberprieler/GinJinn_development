''' Module containing functionality for data set preprocessing.
'''

import json
import os
import shutil
from typing import List
import datetime
import pandas as pd

def flatten_coco(
    ann_file: str,
    img_root_dir: str,
    out_dir: str,
    sep: str = '~',
    custom_id: bool = False,
    annotated_only: bool = False,
    link_images: bool = True
):
    '''flatten_coco

    Flatten COCO data set in such a way that all images are located in the same
    directory.

    Parameters
    ----------
    ann_file : str
        Path to annotation (JSON) file.
    img_root_dir : str
        Root dir of the image directory. For COCO, this directory is often called
        "images".
    out_dir : str
        Output directory.
    sep : str, optional
        Seperator for path flattening, by default '~'
    custom_id : bool, optional
        Whether the new image name should be replaced with a custom id, by default False
    annotated_only : bool, optional
        Whether only annotated images should be kept in the data set.
    link_images : bool, optional
        If true, images won't be copied but hard-linked instead.
    '''
    with open(ann_file) as ann_f:
        annotations = json.load(ann_f)

    out_img_dir = os.path.join(out_dir, 'images')
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)

    if annotated_only:
        img_ids = {ann['image_id'] for ann in annotations['annotations']}
        annotations['images'] = [ann for ann in annotations['images'] if ann['id'] in img_ids]

    id_map = {}

    for i, img_ann in enumerate(annotations['images']):
        file_name = img_ann['file_name']

        if custom_id:
            new_file_name = f'{i}.jpg'
            id_map[i] = file_name
        else:
            new_file_name = file_name.replace('/', sep)

        img_ann['file_name'] = new_file_name
        if link_images:
            os.link(
                os.path.join(img_root_dir, file_name),
                os.path.join(out_img_dir, new_file_name)
            )
        else:
            shutil.copy(
                os.path.join(img_root_dir, file_name),
                os.path.join(out_img_dir, new_file_name)
            )

    out_ann_file = os.path.join(out_dir, 'annotations.json')
    with open(out_ann_file, 'w') as ann_f:
        json.dump(annotations, ann_f, indent=2)

    if custom_id:
        id_map_file = os.path.join(out_dir, 'id_map.csv')
        id_map_df = pd.DataFrame.from_records(
            list(id_map.items()),
            columns=['id', 'original_path']
        )
        id_map_df.to_csv(id_map_file, index=False)


def filter_categories_coco(
    ann_file: str,
    out_file: str,
    drop: List[str] = None,
    keep: List[str] = None
):
    """filter_categories_coco

    This function allows to filter object annotations in a COCO dataset according to their
    assigned category. Therefore, either ``drop`` or ``keep`` has to be specified.
    Note that the original IDs referring to annotations, categories and images are preserved,
    and may be non-contiguous in the output dataset.

    Parameters
    ----------
    ann_file : str
        Path to annotation (JSON) file
    out_file : str
        Output file name
    drop : list of str
        If specified, these categories are removed from the dataset.
    keep : list of str
        If specified, only these categories are preserved.

    Raises
    ------
    ValueError
        Raised for unsupported parameter settings.
    """
    if bool(drop) + bool(keep) == 0:
        raise ValueError(
            "Either ``drop`` or ``keep`` has to be specified as non-empty list."
        )

    with open(ann_file, "rb") as f:
        anno = json.load(f)

    if keep:
        categories_left = {cat["id"] for cat in anno.get("categories") if cat["name"] in keep}
    else:
        categories_left = {cat["id"] for cat in anno.get("categories") if cat["name"] not in drop}

    anno["annotations"] = [ann for ann in anno.get("annotations") if ann["category_id"] in categories_left]
    anno["categories"] = [cat for cat in anno.get("categories") if cat["id"] in categories_left]

    images_left = {ann["image_id"] for ann in anno.get("annotations")}
    anno["images"] = [img for img in anno.get("images") if img["id"] in images_left]

    anno["info"] = {
        "contributor" : "",
        "date_created" : datetime.datetime.now().strftime("%Y/%m/%d"),
        "description" : "",
        "version" : "",
        "url" : "",
        "year" : ""
    }

    # write COCO json file
    with open(out_file, 'w') as json_file:
        json.dump(
            anno,
            json_file,
            indent = 2,
            sort_keys = True
        )
