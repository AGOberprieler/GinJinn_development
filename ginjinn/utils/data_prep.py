''' Module containing functionality for data set preprocessing.
'''

import json
import os
import shutil
import pandas as pd

def flatten_coco(
    ann_file: str,
    img_root_dir: str,
    out_dir: str,
    sep: str = '~',
    custom_id: bool = False,
    annotated_only: bool = False,
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
        # os.symlink(
        #     os.path.join(img_root_dir, file_name),
        #     os.path.join(out_img_dir, new_file_name)
        # )
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
            id_map.items(),
            columns=['id', 'original_path']
        )
        id_map_df.to_csv(id_map_file, index=False)
