'''Module for merging sliding-window cropped datasets
'''

import os
import copy
import shutil
import json
import itertools
from typing import List, Tuple, Callable

import numpy as np
import cv2

from ginjinn.simulation import coco_utils
from ginjinn.utils import load_coco_ann, get_obj_anns

def get_bname_from_fname(file_name: str) -> str:
    '''get_bname_from_fname

    Get original image name from sliding-window cropped file name.

    Parameters
    ----------
    file_name : str
        sliding-window cropped image file name

    Returns
    -------
    str
        original image name sans extension
    '''
    fname, _ = os.path.splitext(file_name)
    fname_split = fname.split('_')
    return '_'.join(fname_split[:-2])

def get_coords_from_fname(file_name: str) -> np.ndarray:
    '''get_coords_from_fname

    Get bounding-box coordinates on the original image of a sliding-window
    crop from file name.

    Parameters
    ----------
    file_name : str
        sliding-window cropped image file name

    Returns
    -------
    np.ndarray
        numpy array of bounding box coordinates on original image (x0, x1, y0, y1).
    '''
    fname, _ = os.path.splitext(file_name)
    fname_split = fname.split('_')
    y0, y1 = [int(coord) for coord in fname_split[-2].split('-')]
    x0, x1 = [int(coord) for coord in fname_split[-1].split('-')]

    return np.array([x0, x1, y0, y1])

def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    '''xywh_to_xyxy

    Translate bounding box format from x0y0wh to x0y0x1y1

    Parameters
    ----------
    xywh : np.ndarray
        bbox in x0y0wh format

    Returns
    -------
    np.ndarray
        bbox in x0y0x1y1 format
    '''
    xyxy = np.array(xywh).reshape(-1, 4)
    xyxy[:,2:4] = xyxy[:,0:2] + xyxy[:,2:4]
    return xyxy

def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    '''xywh_to_xyxy

    Translate bounding box format from x0y0x1y1 to x0y0wh

    Parameters
    ----------
    xyxy : np.ndarray
        bbox in x0y0x1y1 format

    Returns
    -------
    np.ndarray
        bbox in x0y0wh format
    '''
    xywh = np.array(xyxy).reshape(-1, 4)
    xywh[:,2:4] = xywh[:,2:4] - xywh[:,0:2]
    return xywh

def intersection_bboxes(
    a: np.ndarray,
    b: np.ndarray,
    intersection_type: str='iou'
) -> float:
    '''intersection_bboxes

    Calculate intersection of two bounding-boxes a and b.

    Parameters
    ----------
    a : np.ndarray
        Bounding-box in x0y0x1y1 format.
    b : np.ndarray
        Bounding-box in x0y0x1y1 format.
    intersection_type : str, optional
        Type or intersection to calculate, by default 'iou'.

        Possible types are:
        - "simple": absolute intersection area
        - "iou": intersection over union (intersection area / union area)
        - "ios": intersection over smaller (intersection area / smaller bbox area)

    Returns
    -------
    float
        Intersection

    Raises
    ------
    Exception
        Raised when an invalid intersection type is passed.
    '''
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])

    if (dx>=0) and (dy>=0):
        intersection = dx*dy
        if intersection_type == 'simple':
            return intersection
        elif intersection_type == 'iou':
            w = max(a[2], b[2]) - min(a[0], b[0])
            h = max(a[3], b[3]) - min(a[1], b[1])

            return intersection / (w * h)
        elif intersection_type == 'ios':
            w = min(a[2]-a[0], b[2]-b[0])
            h = min(a[3]-a[1], b[3]-b[1])

            return intersection / (w * h)
        else:
            msg = f'Invalid intersection_type argument "{intersection_type}". ' +\
                'Available arguments are "simple", "iou", "ios".'
            raise Exception(msg)

    return 0.0

def intersection_bboxes_coco(
    a: np.ndarray,
    b: np.ndarray,
    intersection_type: str='iou',
):
    '''intersection_bboxes_coco

    Calculate intersection of two COCO bounding-boxes a and b.

    Parameters
    ----------
    a : np.ndarray
        Bounding-box in x0y0wh format.
    b : np.ndarray
        Bounding-box in x0y0wh format.
    intersection_type : str, optional
        Type or intersection to calculate, by default 'iou'.

        Possible types are:
        - "simple": absolute intersection area
        - "iou": intersection over union (intersection area / union area)
        - "ios": intersection over smaller (intersection area / smaller bbox area)

    Returns
    -------
    float
        Intersection
    '''
    # a = [a[0], a[1], a[0]+a[2], a[1]+a[3]]
    # b = [b[0], b[1], b[0]+b[2], b[1]+b[3]]
    a = xywh_to_xyxy(a)
    b = xywh_to_xyxy(b)

    return intersection_bboxes(a, b, intersection_type=intersection_type)

def calc_intersection_matrix(
    bboxes: np.ndarray,
    intersection_type: str='iou'
) -> np.ndarray:
    '''calc_intersection_matrix

    Calculate pair-wise intersection matrix for bounding-boxes.

    Parameters
    ----------
    bboxes : np.ndarray
        n * 4 np.array of bounding boxes in x0y0x1y1 format.
        Each row represents a single bounding box.
    intersection_type : str, optional
        Type or intersection to calculate, by default 'iou'.

        Possible types are:
        - "simple": absolute intersection area
        - "iou": intersection over union (intersection area / union area)
        - "ios": intersection over smaller (intersection area / smaller bbox area)

    Returns
    -------
    np.ndarray
        n * n matrix of pairwise intersections.
    '''
    intersection_matrix = np.ones((bboxes.shape[0], bboxes.shape[0]))
    for i in range(0, bboxes.shape[0]-1):
        for j in range(i + 1, bboxes.shape[0]):
            intersection_matrix[i, j] = intersection_bboxes(
                bboxes[i], bboxes[j],
                intersection_type=intersection_type
            )
            intersection_matrix[j, i] = intersection_matrix[i, j]

    return intersection_matrix

def reconstruct_original_image(
    img_anns: List[dict],
    img_dir: str,
) -> np.ndarray:
    '''reconstruct_original_image

    Reconstruct the original image from cropped sub images.

    Parameters
    ----------
    img_anns : List[dict]
        List of COCO image annotations as dictionary.
    img_dir : str
        Directory containing the images corresponding to img_anns.

    Returns
    -------
    np.ndarray
        RGB image as numpy array (h * w * 3)
    '''
    coords = np.array([get_coords_from_fname(img_ann['file_name']) for img_ann in img_anns])

    orig_h, orig_w = coords.max(0)[[1, 3]]
    orig_img = np.zeros((orig_h, orig_w, 3), dtype=np.int)

    for img_ann in img_anns:
        sub_img = cv2.imread(os.path.join(img_dir, img_ann['file_name']))

        xxyy = get_coords_from_fname(img_ann['file_name'])
        orig_img[xxyy[0]:xxyy[1], xxyy[2]:xxyy[3],:] = sub_img

    return orig_img

def reconstruct_annotations_on_original(
    img_anns: List[dict],
    obj_anns: List[dict],
) -> List[dict]:
    '''reconstruct_annotations_on_original

    Reconstruct object annotations in the coordinate system of
    the original, sliding-window croppped image.

    Parameters
    ----------
    img_anns : List[dict]
        List of COCO image annotations as dictionary.
    obj_anns : List[dict]
        List of COCO object annotations as dictionary.

    Returns
    -------
    List[dict]
        List of COCO image annotations in the coordinate system of
        the original image as dictionary.
    '''
    orig_obj_anns = []
    for img_ann in img_anns:
        xxyy = get_coords_from_fname(img_ann['file_name'])

        sub_obj_anns = [obj_ann for obj_ann in obj_anns if obj_ann['image_id'] == img_ann['id']]
        for obj_ann in sub_obj_anns:
            orig_obj_ann = copy.deepcopy(obj_ann)
            orig_obj_ann['bbox'][0] = list(obj_ann['bbox'][0] + xxyy[2])
            orig_obj_ann['bbox'][1] = list(obj_ann['bbox'][1] + xxyy[0])
            orig_obj_ann['image_id'] = img_anns[0]['id']

            orig_obj_anns.append(orig_obj_ann)

    return orig_obj_anns

def merge_bbox_annotations(
    obj_anns: List[dict],
    img_id: int,
    intersection_type: str='iou',
    intersection_th: float=0.6,
) -> List[dict]:
    '''merge_bbox_annotations

    Merge bounding-box annotations using single-linkage hierarchical
    clustering based on pairwise intersections.

    Parameters
    ----------
    obj_anns : List[dict]
        List of COCO object annotations as dictionary.
    img_id : int
        Image ID merged object annotations should refer to.
    intersection_type : str, optional
        Type or intersection to calculate, by default 'iou'.

        Possible types are:
        - "simple": absolute intersection area
        - "iou": intersection over union (intersection area / union area)
        - "ios": intersection over smaller (intersection area / smaller bbox area)
    intersection_th : float, optional
        Intersection threshold for the clustering cut-off, by default 0.6

    Returns
    -------
    List[dict]
        List of COCO object annotations as dictionary.
    '''
    from sklearn.cluster import AgglomerativeClustering

    if len(obj_anns) < 1:
        return []

    bboxes = xywh_to_xyxy([o_ann['bbox'] for o_ann in obj_anns])
    intersection_matrix = calc_intersection_matrix(bboxes, intersection_type=intersection_type)
    if intersection_matrix.shape[0] < 2:
        cl = np.array([0], dtype=int)
    else:
        ac = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1-intersection_th,
            affinity='precomputed',
            linkage='single'
        )
        cl = ac.fit_predict(1 - intersection_matrix)

    new_anns = []
    for cl_id in np.unique(cl):
        cl_idcs = np.argwhere(cl == cl_id).flatten()
        bboxes_cl = bboxes[cl == cl_id]
        bbox_merged = np.array([*bboxes_cl.min(0)[:2], *bboxes_cl.max(0)[2:]])

        new_ann = copy.deepcopy(obj_anns[cl_idcs[0]])
        bbox_xywh = xyxy_to_xywh(bbox_merged).flatten()
        new_ann['bbox'] = list(bbox_xywh)
        new_ann['area'] = bbox_xywh[2] * bbox_xywh[3]
        new_ann['image_id'] = img_id
        new_anns.append(new_ann)

    return new_anns

def merge_cropped_predictions(
    img_anns: List[dict],
    obj_anns: List[dict],
    img_dir: str,
    intersection_type: str = 'ios',
    intersection_th: float = 0.60,
) -> Tuple[np.ndarray, List[dict], List[dict], List[dict]]:
    '''merge_cropped_predictions

    Merge (sliding-window) cropped COCO annotations using hierarchical,
    single-linkage clustering based on pair-wise bounding-box intersections.

    Parameters
    ----------
    img_anns : List[dict]
        List of COCO image annotations as dictionary.
    obj_anns : List[dict]
        List of COCO object annotations as dictionary.
    img_dir : str
        Directory containing the images, img_anns refer to.
    intersection_type : str, optional
        Type or intersection to calculate, by default 'ios'.

        Possible types are:
        - "simple": absolute intersection area
        - "iou": intersection over union (intersection area / union area)
        - "ios": intersection over smaller (intersection area / smaller bbox area)
    intersection_th : float, optional
        Intersection threshold for the clustering cut-off, by default 0.6

    Returns
    -------
    Tuple[np.ndarray, List[dict], List[dict], List[dict]]
        Tuple containing the reconstructed original image as np.ndarray,
        a new COCO annotation dict for the reconstructed original image,
        the merged  annotations in the coordinate system of the original image,
        and unmerged annotations in the coordinate system of the original image.
        i.e.: (orig_img, orig_img_ann, merged_obj_anns, orig_obj_anns)
    '''
    orig_img = reconstruct_original_image(img_anns, img_dir)
    orig_obj_anns = reconstruct_annotations_on_original(img_anns, obj_anns)
    orig_img_ann = coco_utils.build_coco_image(
        image_id=img_anns[0]['id'],
        file_name=f'{get_bname_from_fname(img_anns[0]["file_name"])}.jpg',
        width=orig_img.shape[1],
        height=orig_img.shape[0],
        coco_url=img_anns[0].get('coco_url', ''),
        date_captured=img_anns[0].get('date_captured', 0),
        flickr_url=img_anns[0].get('flickr_url', ''),
    )

    merged_obj_anns = merge_bbox_annotations(
        obj_anns=orig_obj_anns,
        img_id=orig_img_ann['id'],
        intersection_type=intersection_type,
        intersection_th=intersection_th,
    )

    return orig_img, orig_img_ann, merged_obj_anns, orig_obj_anns

def merge_sliding_window_predictions(
    img_dir: str,
    ann_path: str,
    out_dir: str,
    intersection_type: str = 'ios',
    intersection_th: float = 0.5,
    on_out_dir_exists: Callable[[str], bool] = lambda out_dir: True,
    on_img_out_dir_exists: Callable[[str], bool] = lambda img_out_dir: True,
):
    '''merge_sliding_window_predictions

    Merge predicted annotations that were based on sliding-window cropped images.

    Parameters
    ----------
    img_dir : str
        Path to directory containing the images that the prediction was made for.
    ann_path : str
        Path to predicted annotation.
    out_dir : str
        Path to directory that the merged annotations and images should be written to.
    intersection_type : str, optional
        Type or intersection to calculate, by default 'ios'.

        Possible types are:
        - "simple": absolute intersection area
        - "iou": intersection over union (intersection area / union area)
        - "ios": intersection over smaller (intersection area / smaller bbox area)
    intersection_th : float, optional
        Intersection threshold for the clustering cut-off, by default 0.6
    on_out_dir_exists : Callable[[str], bool], optional
        A function that decides what should happen if out_dir already exists.
        If true is returned, out_dir will be overwritten.
        By default, out_dir will be overwritten.
    on_img_out_dir_exists : Callable[[str], bool], optional
        A function that decides what should happen if img_out_dir already exists.
        If true is returned, img_out_dir will be overwritten.
        By default, img_out_dir will be overwritten.
    '''
    ann = load_coco_ann(ann_path)
    bnames = list({get_bname_from_fname(img_ann['file_name']) for img_ann in ann['images']})

    if os.path.exists(out_dir):
        should_remove = on_out_dir_exists(out_dir)
        if should_remove:
            shutil.rmtree(out_dir)
            os.mkdir(out_dir)
    else:
        os.mkdir(out_dir)

    img_out_dir = os.path.join(out_dir, 'images')
    if os.path.exists(img_out_dir):
        should_remove = on_img_out_dir_exists(img_out_dir)
        if should_remove:
            shutil.rmtree(img_out_dir)
            os.mkdir(img_out_dir)
    else:
        os.mkdir(img_out_dir)

    ann_out_file = os.path.join(out_dir, 'annotations.json')

    new_img_anns = []
    new_obj_anns = []

    for bname in bnames:
        img_anns = [
            img_ann
            for img_ann in ann['images'] if get_bname_from_fname(img_ann['file_name']) == bname
        ]
        obj_anns = list(itertools.chain.from_iterable(
            [get_obj_anns(img_ann, ann) for img_ann in img_anns]
        ))

        orig_img, orig_img_ann, merged_obj_anns, _ = merge_cropped_predictions(
            img_anns,
            obj_anns,
            img_dir,
            intersection_type=intersection_type,
            intersection_th=intersection_th
        )

        img_f = os.path.join(img_out_dir, f'{bname}.jpg')
        cv2.imwrite(img_f, orig_img)

        new_img_anns.append(orig_img_ann)
        new_obj_anns.extend(merged_obj_anns)

    new_ann = coco_utils.build_coco_dataset(
        annotations=new_obj_anns,
        images=new_img_anns,
        categories=ann['categories'],
        licenses=ann['licenses'],
        info=ann['info']
    )

    with open(ann_out_file, 'w') as ann_f:
        json.dump(new_ann, ann_f, indent=2)
