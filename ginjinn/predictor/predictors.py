''' Prediction module
'''

import glob
import json
import os
from datetime import datetime
from typing import List
import cv2
import imantics
import numpy as np
from detectron2.engine.defaults import DefaultPredictor


class GinjinnPredictor(DefaultPredictor):
    '''GinjinnPredictor
        A class for predicting from a trained Detectron2 model.

        Parameters
        ----------
        cfg : Object
            Detectron2 configuration object
        replicates : int, optional
            Number of replicated predictions to conduct for each image,
            by default 1
    '''
    def __init__(
        self,
        cfg,
        replicates: int = 1
    ):
        super().__init__(cfg)
        self.replicates = replicates # TODO: thin about whether we need this


def predict_and_save(
    img_dir: str,
    outdir: str,
    class_names: List[str],
    task: str,
    predictor: object,
    save_cropped: bool = True,
    threshold: float = 0,
    crop_margin: int = 0,
    img_names: List[str] = None
):
    """
    This function applies a predictor to multiple input image and stores the results
    as COCO json file (segmentations encoded as polygons). Optionally, bounding boxes
    for each image can be cropped from the input images and segmentation masks.

    Parameters
    ----------
    img_dir : str
        Directory containing input images
    outdir : str
        Directory for saving prediction results
    class_names : list of str
        Ordered list of object class names
    task : str
        "bbox-detection" or "instance-segmentation"
    predictor : object
        e.g. detecton2's dafult predictor
    save_cropped : bool
        If true, bounding boxes for each predicted instance above threshold are cropped
        and saved as images. In case of instance segmentation, segmentation masks are
        cropped as well.
    threshold : float
        Only predictions with scores >= threshold are saved.
        Should be within [0, 1].
        This has the same effect as using DefaultPredictor(cfg) with corresponding
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    crop_margin : int
        Expands bounding boxes by crop_margin pixels at each side before cropping.
    img_names : list of str
        File names of input images. By Default, all JPG images at the top level of
        img_dir are used.
    """

    # get image names
    if not img_names:
        patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        img_paths = []
        for pat in patterns:
            img_paths.extend(glob.glob(os.path.join(img_dir, pat)))
        img_names = [os.path.split(path)[-1] for path in img_paths]

    # create or clean subdirectories of outdir
    output_subdirs = []

    if save_cropped:
        output_subdirs.append("images_cropped")
        if task == "instance-segmentation":
            output_subdirs.append("masks_cropped")

    for subdir in output_subdirs:
        target_dir = os.path.join(outdir, subdir)
        os.makedirs(target_dir, exist_ok=True)
        for path in glob.iglob(os.path.join(target_dir, "*")):
            os.remove(path)

    # initialize COCO
    info = {
        "contributor" : "",
        "date_created" : datetime.now().strftime("%Y/%m/%d"),
        "description" : "",
        "version" : "",
        "url" : "",
        "year" : ""
    }
    licenses = [{"id": 1, "name": "", "url": ""}]
    categories = [
        {"id": i+1, "name": cl, "supercategory": ""} for (i, cl) in enumerate(class_names)
    ]
    annotations = []
    images = []

    i_ann = 1

    # process images
    for i_img, img_name in enumerate(img_names):

        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        height, width = image.shape[:2]

        images.append({
            "id": i_img + 1,
            "file_name": img_name,
            "coco_url": "",
            "date_captured": "",
            "flickr_url": "",
            "height": height,
            "width": width,
            "license": 1
        })

        predictions = predictor(image)

        # convert to numpy arrays
        scores = predictions["instances"].get_fields()["scores"].to("cpu").numpy()
        classes = predictions["instances"].get_fields()["pred_classes"].to("cpu").numpy()
        boxes = predictions["instances"].get_fields()["pred_boxes"].to("cpu").tensor.numpy()

        if task == "instance-segmentation":
            masks = predictions["instances"].get_fields()["pred_masks"].to("cpu").numpy()

        # process instances
        for i_inst, score in enumerate(scores):
            if score >= threshold:
                bbox = boxes[i_inst]

                if save_cropped:
                    x1, y1, x2, y2 = [round(coord) for coord in bbox]
                    x1, x2 = np.clip((x1 - crop_margin, x2 + crop_margin), 0, width - 1)
                    y1, y2 = np.clip((y1 - crop_margin, y2 + crop_margin), 0, height - 1)
                    image_cropped = image[y1:y2, x1:x2]

                    outpath = os.path.join(
                        outdir,
                        "images_cropped",
                        "img_{}_{}.jpg".format(i_img + 1, i_inst + 1)
                    )
                    cv2.imwrite(outpath, image_cropped)

                    if task == "instance-segmentation":
                        mask_cropped = masks[i_inst][y1:y2, x1:x2]
                        outpath = os.path.join(
                            outdir,
                            "masks_cropped",
                            "mask_{}_{}.png".format(i_img + 1, i_inst + 1)
                        )
                        cv2.imwrite(outpath, mask_cropped.astype("uint8")*255)

                anno = {
                    "area": ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])).tolist(),
                    "bbox": bbox.tolist(),
                    "iscrowd": 0,
                    "image_id": i_img + 1,
                    "id": i_ann,
                    "category_id": classes[i_inst].tolist() + 1
                }

                if task == "instance-segmentation":
                    imask = imantics.Mask(masks[i_inst])
                    ipoly = imask.polygons()
                    anno["segmentation"] = ipoly.segmentation

                annotations.append(anno)
                i_ann += 1

        # write COCO annotation file
        json_new = os.path.join(outdir, "annotations.json")
        with open(json_new, 'w') as json_file:
            json.dump({
                'info': info,
                'licenses': licenses,
                'images': images,
                'annotations': annotations,
                'categories': categories
                },
                json_file,
                indent=2,
                sort_keys=True
            )
