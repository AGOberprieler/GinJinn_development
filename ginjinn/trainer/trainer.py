"""
Classes for training and, optionally, simultaneous validation.
"""

import copy
import datetime
import logging
import os
import time
import torch
import numpy as np
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.engine.defaults import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils import comm


class Trainer(DefaultTrainer):
    """Trainer class which allows to set the applied augmentations at runtime.
    """
    _augmentations = []

    @classmethod
    def set_augmentations(cls, augmentations):
        """Specify augmentations for training.

        Parameters
        ----------
        augmentations : list
            Augmentations to be applied
        """
        cls._augmentations = augmentations

    @classmethod
    def build_train_loader(cls, cfg):
        """Build data loader for training.

        Returns
        ----------
        torch.utils.data.DataLoader
            Data loader
        """
        augs = [T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )]
        augs.extend(cls._augmentations)

        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                is_train=True,
                augmentations=augs
            )
        )

    @classmethod
    def from_ginjinn_config(cls, gj_cfg):
        '''from_ginjinn_config

        Build ValTrainer object from GinjinnConfiguration instead of
        detectron2 configuration.

        Parameters
        ----------
        gj_cfg
            GinjinnConfiguration object.
        '''

        cls.set_augmentations(gj_cfg.augmentation.to_detectron2_augmentations())

        detectron2_cfg = gj_cfg.to_detectron2_config()

        return cls(detectron2_cfg)

    ##alternative:
    #@classmethod
    #def build_train_loader(cls, cfg):
        #"""Build data loader for training.

        #Returns
        #----------
        #torch.utils.data.DataLoader
            #Data loader
        #"""
        #augs = [T.ResizeShortestEdge(
            #cfg.INPUT.MIN_SIZE_TRAIN,
            #cfg.INPUT.MAX_SIZE_TRAIN,
            #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        #)]
        #augs.extend(cls._augmentations)

        #return build_detection_train_loader(
            #cfg,
            #mapper = lambda data_dict: mapper_train(data_dict, augs)
        #)



class ValTrainer(Trainer):
    """This trainer class evaluates validation data during training.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Builds COCO evaluator for a given dataset.

        Parameters
        ----------
        dataset_name : str
        output_folder : str

        Returns
        ----------
        DatasetEvaluator
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(
                    self.cfg,
                    is_train=True, # required to obtain losses
                    # no flip
                    augmentations=[T.ResizeShortestEdge(
                        self.cfg.INPUT.MIN_SIZE_TRAIN,
                        self.cfg.INPUT.MAX_SIZE_TRAIN,
                        self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
                    )]
                )
            )
        ))
        return hooks


def mapper_train(dataset_dict, augmentations):
    """
    This basic mapper function takes a dataset dictionary in Detectron2 format,
    and maps it to a format used by the model.

    Parameters
    ----------
    dataset_dict : dict
        Annotations for one image in Detectron2 format
    augmentations : list
        Augmentations and transformations to be applied
    
    Returns
    -------
    dict
        Format accepted by builtin models in Detectron2
    """
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")
    detection_utils.check_image_size(dataset_dict, image)

    image, transforms = T.apply_transform_gens(augmentations, image)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32")) # pylint: disable=E1101

    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
    return dataset_dict


class LossEvalHook(HookBase):
    """
    This hook allows periodic loss calculation for the validation data set.
    It is executed every ``eval_period`` iterations and after the last iteration.
    """
    # pylint: disable=E1101
    def __init__(self, eval_period, model, data_loader):
        """
        Parameters
        ----------
        eval_period : int
            Period to calculate losses. If set to 0, they are only calculated after the
            last iteration.
        model : torch.nn.Module
            Model to be used
        data_loader : iterable
            produces data to be run by `model(data)`
        """
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # see evaluator.py from Detectron2
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses_all = {}
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            losses_batch = self._get_loss(inputs)
            if losses_all:
                for key in losses_batch:
                    losses_all[key].append(losses_batch[key])
            else:
                for key in losses_batch:
                    losses_all[key] = [losses_batch[key]]

        losses_mean = {key + "_val": np.mean(values) for (key, values) in losses_all.items()}
        losses_mean["total_loss_val"] = sum(losses_mean.values())
        self.trainer.storage.put_scalars(**losses_mean)

        comm.synchronize()
        return losses_mean

    def _get_loss(self, data):
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        return metrics_dict

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
