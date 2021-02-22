"""
Module for generating datasets with cropped object instances.
"""

import datetime
import glob
import json
import os
import copy
import xml
from typing import List, Sequence, Tuple, Optional
import numpy as np
import cv2
import imantics
from pycocotools import mask
from ginjinn.simulation import coco_utils
from .utils import load_coco_ann, get_obj_anns
from .utils import get_pvoc_obj_bbox, bbox_from_mask,\
    crop_bbox, bbox_size, set_pvoc_obj_bbox,\
    drop_pvoc_objects, get_pvoc_filename, set_pvoc_filename,\
    get_pvoc_objects, add_pvoc_object, set_pvoc_size, get_pvoc_size,\
    load_pvoc_annotation, write_pvoc_annotation, coco_seg_to_mask

# pylint: disable=C0103
def crop_annotations(
    annotations: List,
    img_width: int,
    img_height: int,
    cropping_range: Sequence[int],
    start_id: int,
    task: str,
    keep_incomplete: bool = True
) -> Tuple:
    """Crop object annotations.

    This function transforms a list of object annotations in COCO format, such that the resulting
    annotations refer to a cropped version of the original image. The output only contains objects
    with valid, non-empty bounding boxes or segmentations, depending on the specified task.

    Parameters
    ----------
    annotations : list of dict
        List of object annotations referring to the same image.
    img_width : int
        (Original) image width
    img_height : int
        (Original) image height
    cropping_range: sequence of int
        (x0, x1, y0, y1) slices segmentation masks by x0:x1 (horizontally) and y0:y1 (vertically).
    start_id : int
        Object ID to start output annotations with.
    task : str
        Either "bbox-detection" or "instance-segmentation"
    keep_incomplete : bool
        If false, trimmed objects are discarded.


    Returns
    -------
    next_id
        This may be useful as start id for further objects.
    annotation_cropped
        List of transformed COCO annotations with non-empty segmentation

    Raises
    ------
    TypeError
        Raised for unsupported object annotations.
    ValueError
        Raised if input annotations refer to different images.
    """
    x_start, x_end, y_start, y_end = cropping_range
    annotations_cropped = []
    img_id = None
    i_ann = start_id

    for annotation in annotations:
        # check image id
        if img_id:
            if annotation["image_id"] != img_id:
                raise ValueError(
                    "All annotations must refer to the same image, i.e., have equal image_id's."
                )
        else:
            img_id = annotation["image_id"]

        if task == "bbox-detection":
            # get bounding box
            bbox_orig = annotation.get("bbox")
            if bbox_orig:
                if not isinstance(bbox_orig, list) or len(bbox_orig) != 4:
                    raise TypeError(
                        "Unknown bbox format, list of length 4 expected."
                    )
            else:
                # skip instance
                continue

            # transform box
            x1, y1, w, h = bbox_orig
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = (round(coord) for coord in (x1, y1, x2, y2))
            X1 = np.clip(x1 - x_start, 0, x_end - x_start - 1).tolist()
            X2 = np.clip(x2 - x_start, 0, x_end - x_start).tolist()
            Y1 = np.clip(y1 - y_start, 0, y_end - y_start - 1).tolist()
            Y2 = np.clip(y2 - y_start, 0, y_end - y_start).tolist()
            area = (X2 - X1) * (Y2 - Y1)

            if area > 0:
                if keep_incomplete or (X2-X1 >= x2-x1 and Y2-Y1 >= y2-y1):
                    # create object annotation
                    annotations_cropped.append({
                        "area": area,
                        "bbox": [X1, Y1, X2-X1, Y2-Y1],
                        "image_id": img_id,
                        "id": i_ann,
                        "category_id": annotation.get("category_id")
                    })
                    i_ann += 1

        elif task == "instance-segmentation":
            # read segmentation
            seg_orig = annotation.get("segmentation")
            if seg_orig:
                mask_orig = coco_seg_to_mask(seg_orig, img_width, img_height)
            else:
                # skip instances without segmentation
                continue

            # crop segmentation
            imask_cropped = imantics.Mask(mask_orig[y_start:y_end, x_start:x_end])
            seg_cropped = imask_cropped.polygons().segmentation

            if seg_cropped:
                # compare object boundaries
                bbox_orig = bbox_from_mask(mask_orig, fmt="xywh").tolist()
                bbox_cropped = bbox_from_mask(imask_cropped.array, fmt="xywh").tolist()

                if (
                    keep_incomplete
                    or (bbox_cropped[2] >= bbox_orig[2] and bbox_cropped[3] >= bbox_orig[3])
                ):
                    # create object annotation
                    annotations_cropped.append({
                        "area": bbox_cropped[2] * bbox_cropped[3],
                        "bbox": bbox_cropped,
                        "segmentation": seg_cropped,
                        "iscrowd": 0,
                        "image_id": img_id,
                        "id": i_ann,
                        "category_id": annotation.get("category_id")
                    })
                    i_ann += 1

    return i_ann, annotations_cropped

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
    padding : int
        This option allows to increase the cropping range beyond the borders of a segmented object.
        If possible, each side of the corresponding bounding box is shifted by the same number of
        pixels.

    Raises
    ------
    TypeError
        Raised for unsupported segmentation format.
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
            x1, x2 = np.clip((x1 - padding, x2 + padding), 0, width - 1).tolist()
            y1, y2 = np.clip((y1 - padding, y2 + padding), 0, height - 1).tolist()

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
            polygons_cropped = [p for p in polygons_cropped if len(p) >= 6]

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

# TODO: this should be reworked and improved
def sliding_window_idcs(l: int, n: int, overlap: float = 0.5) -> Tuple:
    '''sliding_window_idcs

    >EXPERIMENTAL< Generate sliding window start and stop indices.

    Parameters
    ----------
    l : int
        length
    n : int
        number of non-sliding windows
    overlap : float, optional
        overlap between sliding-windows, by default 0.5

    Returns
    -------
    Tuple
        Tuple of start indices and stop indices
    '''
    window = l / n
    stride = window * (1-overlap)
    starts = np.arange(0, l, stride)[:-1].astype(int)
    stops = (starts + window).astype(int)

    return starts, stops

def sliding_window_idcs_2d(
    x: int, y: int,
    n_x: int, n_y: int,
    overlap: float = 0.5,
) -> Tuple:
    '''sliding_window_idcs_2d

    >EXPERIMENTAL< Generate sliding window start and stop indices.

    Parameters
    ----------
    x : int
        length in x
    y : int
        length in y
    n_x : int
        number of non-sliding windows in x
    n_y : int
        number of non-sliding windows in y
    overlap : float, optional
        overlap between sliding-windows, by default 0.5

    Returns
    -------
    Tuple
        Tuple of start and stop indices, i.e. ((start_x, stop_x), (start_y, stop_y)).
    '''
    return (
        sliding_window_idcs(x, n_x, overlap),
        sliding_window_idcs(y, n_y, overlap),
    )

def sliding_window_grid_2d(
    x: int, y: int,
    n_x: int, n_y: int,
    overlap: float = 0.5
):
    '''sliding_window_grid_2d

    >EXPERIMENTAL< Generate sliding window start and stop indices.

    Parameters
    ----------
    x : int
        length in x
    y : int
        length in y
    n_x : int
        number of non-sliding windows in x
    n_y : int
        number of non-sliding windows in y
    overlap : float, optional
        overlap between sliding-windows, by default 0.5

    Returns
    -------
    np.ndarray
        2D numpy array, where each row consists of the four values
        start_x, stop_x, start_y, stop_y.
    '''
    (x_0, x_1), (y_0, y_1) = sliding_window_idcs_2d(
        x, y,
        n_x, n_y, overlap=overlap
    )
    xxyy = np.array([[*x01, *y01] for x01 in zip(x_0, x_1) for y01 in zip(y_0, y_1)])

    return xxyy

def crop_ann_img(
    img,
    img_ann: dict,
    obj_anns: dict,
    xxyy,
    obj_id: int,
    img_id: int,
    task: str = 'instance-segmentation',
    return_empty: bool = True,
) -> Tuple:
    '''crop_ann_img

    <EXPERIMENTAL> Crop sub-images and sub-annotations from an annotated image.

    Parameters
    ----------
    img
        Image as numpy array.
    img_ann : dict
        Image annotation as COCO dict.
    obj_anns : dict
        Object annotations as list of COCO dicts.
    xxyy
        2D numpy array, where each row consists of the four values
        x0, x1, y0, y1 for cropping.
    obj_id : int
        Start object ID for new, cropped COCO object annotations.
    img_id : int
        Start object ID for new, cropped COCO images.
    task : str, optional
        Either 'instance-segmentation' or 'bbox-detection',by default 'instance-segmentation'.
    return_empty : bool, optional
        Whether images without annotation should be returned, by default True

    Yields
    -------
    Tuple
        Tuple of
        (cropped_img, cropped_img_ann, cropped_img_name, cropped_obj_anns, img_id, obj_id).
    '''
    for cropping_range in xxyy:
        obj_id, cropped_obj_anns = crop_annotations(
            annotations=obj_anns,
            img_width=img.shape[1],
            img_height=img.shape[0],
            cropping_range=list(cropping_range),
            start_id=obj_id,
            task=task,
        )
        if not return_empty:
            if len(cropped_obj_anns) < 1:
                continue

        cropped_img = img[
            cropping_range[2]:cropping_range[3],
            cropping_range[0]:cropping_range[1],
        ]

        for ann in cropped_obj_anns:
            ann['image_id'] = img_id

        img_name = os.path.basename(img_ann['file_name']).split('.')[0]
        # think about whether the name should contain the upper range
        # inclusively or exclusively
        cropped_img_name = '{}_{}-{}_{}-{}.jpg'.format(
            img_name,
            cropping_range[0], cropping_range[1],
            cropping_range[2], cropping_range[3]
        )
        cropped_img_ann = coco_utils.build_coco_image(
            image_id = img_id,
            file_name = cropped_img_name,
            width = cropped_img.shape[1],
            height = cropped_img.shape[0],
            license = img_ann.get('license', 0),
            coco_url = img_ann.get('coco_url', ''),
            date_captured = img_ann.get('date_captured', 0),
            flickr_url = img_ann.get('flickr_url', ''),
        )

        img_id = img_id + 1
        yield (
            cropped_img,
            cropped_img_ann,
            cropped_img_name,
            cropped_obj_anns,
            img_id,
            obj_id
        )

def sliding_window_crop_coco(
    img_dir: str,
    ann_path: str,
    img_dir_out: str,
    ann_path_out: str,
    n_x: int,
    n_y: int,
    overlap: float,
    img_id: int = 0,
    obj_id: int = 0,
    save_empty=True,
):
    '''sliding_window_crop_coco

    >Experimental< Crop sliding window subimages and corresponding
    annotations from COCO annotated images.

    Parameters
    ----------
    img_dir : str
        Image directory.
    ann_path : str
        COCO annotation path.
    img_dir_out : str
        Output directory for images.
    ann_path_out : str
        Output path for COCO annotation.
    n_x : int
        number of non-sliding windows in x
    n_y : int
        number of non-sliding windows in y
    overlap : float, optional
        overlap between sliding-windows, by default 0.5
    img_id : int, optional
        Start image ID for new COCO images, by default 0
    obj_id : int, optional
        Start image ID for new COCO object annotations, by default 0
    save_empty : bool, optional
        Whether images without annotations should be saved, by default True
    '''
    ann = load_coco_ann(ann_path)
    img_anns = ann['images']

    new_obj_anns = []
    new_img_anns = []

    for img_ann in img_anns:
        img = cv2.imread(os.path.join(img_dir, img_ann['file_name']))
        obj_anns = get_obj_anns(img_ann, ann)

        xxyy = sliding_window_grid_2d(
            img.shape[1], img.shape[0],
            n_x, n_y, overlap=overlap
        )

        i_id, o_id = img_id, obj_id
        for c_img, c_img_ann, c_img_name, c_obj_anns, i_id, o_id in crop_ann_img(
            img=img,
            img_ann=img_ann,
            obj_anns=obj_anns,
            xxyy=xxyy,
            obj_id=obj_id,
            img_id=img_id,
            return_empty=save_empty
        ):
            new_img_anns.append(c_img_ann)
            new_obj_anns.extend(c_obj_anns)

            cv2.imwrite(
                os.path.join(img_dir_out, c_img_name),
                c_img,
            )

        img_id, obj_id = i_id, o_id

    new_ann = coco_utils.build_coco_dataset(
        annotations = new_obj_anns,
        images = new_img_anns,
        categories = ann['categories'],
        licenses = ann['licenses'],
        info = ann['info'],
    )

    with open(ann_path_out, 'w') as ann_f:
        json.dump(new_ann, ann_f)

def crop_pvoc_obj(
    obj: xml.etree.ElementTree.ElementTree,
    cropping_range: Sequence[int],
    min_size: Sequence[int] = [10, 10],
) -> Optional[xml.etree.ElementTree.ElementTree]:
    '''crop_pvoc_obj

    Crop PVOC object to specified range.

    Parameters
    ----------
    obj : xml.etree.ElementTree.ElementTree
        PVOC object as ElementTree
    cropping_range : Sequence[int]
        Cropping range in x0x1y0y1 format.
    min_size : Sequence[int], optional
        Minimum cropped bounding-box size (width, height),
        by default [10, 10].

    Returns
    -------
    Optional[xml.etree.ElementTree.ElementTree]
        Cropped PVOC object, or None if the cropped object is
        smaller than min_size.
    '''
    cropped_obj = copy.deepcopy(obj)

    bbox = get_pvoc_obj_bbox(obj)
    bbox_cropped = [int(x) for x in crop_bbox(bbox, cropping_range)]
    w, h = bbox_size(bbox_cropped)

    if w < min_size[0] or h < min_size[1]:
        return None

    set_pvoc_obj_bbox(cropped_obj, bbox_cropped)

    return cropped_obj

def crop_pvoc_ann(
    ann: xml.etree.ElementTree.ElementTree,
    cropping_range: Sequence[int],
    min_size: Sequence[int] = [10, 10],
    rename: bool = True,
) -> xml.etree.ElementTree.ElementTree:
    '''crop_pvoc_ann

    Crop PVOC annotation to specified range.

    Parameters
    ----------
    ann : xml.etree.ElementTree.ElementTree
        PVOC annotation as ElementTree
    cropping_range : Sequence[int]
        Cropping range in x0x1y0y1 format.
    min_size : Sequence[int], optional
        Minimum cropped bounding-box size (width, height),
        by default [10, 10].
    rename : bool, optional
        Whether the filename element text should be change to include
        the cropping coordinates, by default True

    Returns
    -------
    xml.etree.ElementTree.ElementTree
        Cropped PVOC annotation as ElementTree
    '''
    cropped_ann = copy.deepcopy(ann)
    drop_pvoc_objects(cropped_ann)

    if rename:
        nm, ext = os.path.splitext(
            os.path.basename(get_pvoc_filename(ann))
        )
        xmn, xmx, ymn, ymx = cropping_range
        new_name = f'{nm}_{xmn}-{xmx}_{ymn}-{ymx}{ext}'
        set_pvoc_filename(cropped_ann, new_name)

    cropped_objs = [
        obj for obj in [
            crop_pvoc_obj(
                obj, cropping_range, min_size
            ) for obj in get_pvoc_objects(ann)
        ] if not obj is None
    ]

    for obj in cropped_objs:
        add_pvoc_object(cropped_ann, obj)
    set_pvoc_size(
        cropped_ann,
        [
            cropping_range[1] - cropping_range[0],
            cropping_range[3] - cropping_range[2],
            get_pvoc_size(ann)[2],
        ]
    )

    return cropped_ann

def crop_image(
    img: np.ndarray,
    cropping_range: Sequence[int],
) -> np.ndarray:
    '''crop_image

    Crop image to specified range.

    Parameters
    ----------
    img : np.ndarray
        Image as numpy array.
    cropping_range : Sequence[int]
        Cropping range in x0x1y0y1 format.

    Returns
    -------
    np.ndarray
        Cropped image as numpy array.
    '''
    return img[
        cropping_range[2]:cropping_range[3],
        cropping_range[0]:cropping_range[1],
    ]

def sliding_window_crop_pvoc(
    img_dir: str,
    ann_dir: str,
    img_dir_out: str,
    ann_dir_out: str,
    n_x: int,
    n_y: int,
    overlap: float,
    save_empty=True,
):
    '''sliding_window_crop_pvoc

    Sliding-window crop images and annotation from
    PVOC annotated images.

    Parameters
    ----------
    img_dir : str
        Path to image directory
    ann_dir : str
        Path to annotation directory
    img_dir_out : str
        Path to output image directory
    ann_dir_out : str
        Path to output annotation directory
    n_x : int
        number of non-sliding windows in x
    n_y : int
        number of non-sliding windows in y
    overlap : float, optional
        overlap between sliding-windows, by default 0.5
    save_empty : bool, optional
        Whether cropped images without object annotations should be saved,
        by default True
    '''
    ann_files = glob.glob(os.path.join(ann_dir, '*.xml'))

    for ann_f in ann_files:
        ann = load_pvoc_annotation(ann_f)
        img = cv2.imread(os.path.join(img_dir, get_pvoc_filename(ann)))

        xxyy = np.round(sliding_window_grid_2d(
            img.shape[1], img.shape[0],
            n_x, n_y, overlap=overlap
        )).astype(int)

        print(get_pvoc_filename(ann))

        for cropping_range in xxyy:
            cropped_ann = crop_pvoc_ann(ann, cropping_range)

            if not save_empty and len(get_pvoc_objects(cropped_ann)) < 1:
                continue

            cropped_img = crop_image(img, cropping_range)

            filename, ext = os.path.splitext(get_pvoc_filename(cropped_ann))
            ann_filepath = os.path.join(ann_dir_out, f'{filename}.xml')
            write_pvoc_annotation(
                cropped_ann,
                ann_filepath
            )
            img_filepath = os.path.join(img_dir_out, f'{filename}{ext}')
            cv2.imwrite(img_filepath, cropped_img)
