"""
Module for generating datasets with cropped object instances.
"""

import datetime
import glob
import json
import os
import numpy as np
import cv2
import imantics
from pycocotools import mask


# pylint: disable=C0103
def crop_seg_from_coco(
    ann_file: str,
    img_dir: str,
    outdir: str,
    padding: int = 0
):
    """
    This function reads annotations in COCO format and crops each segmentation instance from
    the corresponding image file. In addition, a new COCO json file is written, which annotates
    the cropped images.

    Parameters
    ----------
    ann_file : str
        COCO json file
    img_dir : str
        Directory containing JPG images
    outdir : str
        Directory to which the output is written
    padding : int, default=0
        This option allows to increase the cropping range beyond the borders of a segmented object.
        If possible, each side of the corresponding bounding box is shifted by the same number of
        pixels.
    """

    os.makedirs(os.path.join(outdir, "images_cropped"), exist_ok=True)
    for path in glob.iglob(os.path.join(outdir, "images_cropped", "*")):
        os.remove(path)

    info = {
        "contributor" : "",
        "date_created" : datetime.datetime.now().strftime("%Y/%m/%d"),
        "description" : "",
        "version" : "",
        "url" : "",
        "year" : ""
    }

    # image id -> COCO dict of uncropped image
    dict_images = dict()

    annotations = []
    images = []

    with open(ann_file, "rb") as f:
        ann = json.load(f)

        categories = ann.get("categories")
        licenses = ann.get("licenses")

        for image in ann.get("images"):
            dict_images[image["id"]] = image

        i_ann = 1
        for annotation in ann.get("annotations"):
            img_coco = dict_images[annotation["image_id"]]

            # read image
            img_name = os.path.split(img_coco["file_name"])[1]
            image = cv2.imread(os.path.join(img_dir, img_name))
            # original size
            height = image.shape[0]
            width = image.shape[1]

            seg = annotation.get("segmentation")
            if seg:
                if isinstance(seg, dict):
                    # rle to mask
                    seg_mask = mask.decode(seg).astype("bool")
                elif isinstance(seg, list):
                    # polygon to mask
                    polygons = imantics.Polygons(seg)
                    seg_mask = polygons.mask(width, height).array
                else:
                    raise TypeError(
                        "Unknown segmentation format, polygons or RLE expected"
                    )
            else:
                # skip instances without segmentation
                continue

            # calculate bounding box from segmentation
            x_any = seg_mask.any(axis=0)
            y_any = seg_mask.any(axis=1)
            x = np.where(x_any == True)[0]
            y = np.where(y_any == True)[0]
            if len(x) > 0 and len(y) > 0:
                bbox = [x[0], y[0], x[-1] + 1, y[-1] + 1]
            else:
                continue

            # apply padding, clip values
            x1, y1, x2, y2 = (round(coord) for coord in bbox)
            x1, x2 = np.clip((x1 - padding, x2 + padding), 0, width).tolist()
            y1, y2 = np.clip((y1 - padding, y2 + padding), 0, height).tolist()

            # crop image
            image_cropped = image[y1:y2, x1:x2]
            if image_cropped.size == 0:
                continue

            outpath = os.path.join(
                outdir,
                "images_cropped",
                "img_{}.jpg".format(i_ann)
            )
            cv2.imwrite(outpath, image_cropped)

            images.append({
                "id": i_ann,
                "file_name": "img_{}.jpg".format(i_ann),
                "height": y2 - y1,
                "width": x2 - x1,
                "license": img_coco.get("license")
            })

            # annotate cropped instance
            mask_cropped = seg_mask[y1:y2, x1:x2]

            x_any = mask_cropped.any(axis=0)
            y_any = mask_cropped.any(axis=1)
            x = np.where(x_any == True)[0].tolist()
            y = np.where(y_any == True)[0].tolist()
            x1, y1, x2, y2 = (x[0], y[0], x[-1] + 1, y[-1] + 1)
            bbox_coco = [ x1, y1, x2 - x1, y2 - y1 ]

            polygons_cropped = imantics.Mask(mask_cropped).polygons().segmentation

            annotations.append({
                "area": (x2 - x1) * (y2 - y1),
                "bbox": bbox_coco,
                "segmentation": polygons_cropped,
                "iscrowd": 0,
                "image_id": i_ann,
                "id": i_ann,
                "category_id": annotation["category_id"]
            })

            i_ann += 1

    # write COCO json file
    json_new = os.path.join(outdir, "annotations_cropped.json")
    with open(json_new, 'w') as json_file:
        json.dump({
            'info': info,
            'licenses': licenses,
            'images': images,
            'annotations': annotations,
            'categories': categories
            },
            json_file,
            indent = 2,
            sort_keys = True
        )
