# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by AGOberprieler, 2020, using modifications made by Marcelo Ortega:
# https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
#
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
# 1. Definitions.
#
# "License" shall mean the terms and conditions for use, reproduction,
# and distribution as defined by Sections 1 through 9 of this document.
#
# "Licensor" shall mean the copyright owner or entity authorized by
# the copyright owner that is granting the License.
#
# "Legal Entity" shall mean the union of the acting entity and all
# other entities that control, are controlled by, or are under common
# control with that entity. For the purposes of this definition,
# "control" means (i) the power, direct or indirect, to cause the
# direction or management of such entity, whether by contract or
# otherwise, or (ii) ownership of fifty percent (50%) or more of the
# outstanding shares, or (iii) beneficial ownership of such entity.
#
# "You" (or "Your") shall mean an individual or Legal Entity
# exercising permissions granted by this License.
#
# "Source" form shall mean the preferred form for making modifications,
# including but not limited to software source code, documentation
# source, and configuration files.
#
# "Object" form shall mean any form resulting from mechanical
# transformation or translation of a Source form, including but
# not limited to compiled object code, generated documentation,
# and conversions to other media types.
#
# "Work" shall mean the work of authorship, whether in Source or
# Object form, made available under the License, as indicated by a
# copyright notice that is included in or attached to the work
# (an example is provided in the Appendix below).
#
# "Derivative Works" shall mean any work, whether in Source or Object
# form, that is based on (or derived from) the Work and for which the
# editorial revisions, annotations, elaborations, or other modifications
# represent, as a whole, an original work of authorship. For the purposes
# of this License, Derivative Works shall not include works that remain
# separable from, or merely link (or bind by name) to the interfaces of,
# the Work and Derivative Works thereof.
#
# "Contribution" shall mean any work of authorship, including
# the original version of the Work and any modifications or additions
# to that Work or Derivative Works thereof, that is intentionally
# submitted to Licensor for inclusion in the Work by the copyright owner
# or by an individual or Legal Entity authorized to submit on behalf of
# the copyright owner. For the purposes of this definition, "submitted"
# means any form of electronic, verbal, or written communication sent
# to the Licensor or its representatives, including but not limited to
# communication on electronic mailing lists, source code control systems,
# and issue tracking systems that are managed by, or on behalf of, the
# Licensor for the purpose of discussing and improving the Work, but
# excluding communication that is conspicuously marked or otherwise
# designated in writing by the copyright owner as "Not a Contribution."
#
# "Contributor" shall mean Licensor and any individual or Legal Entity
# on behalf of whom a Contribution has been received by Licensor and
# subsequently incorporated within the Work.
#
# 2. Grant of Copyright License. Subject to the terms and conditions of
# this License, each Contributor hereby grants to You a perpetual,
# worldwide, non-exclusive, no-charge, royalty-free, irrevocable
# copyright license to reproduce, prepare Derivative Works of,
# publicly display, publicly perform, sublicense, and distribute the
# Work and such Derivative Works in Source or Object form.
#
# 3. Grant of Patent License. Subject to the terms and conditions of
# this License, each Contributor hereby grants to You a perpetual,
# worldwide, non-exclusive, no-charge, royalty-free, irrevocable
# (except as stated in this section) patent license to make, have made,
# use, offer to sell, sell, import, and otherwise transfer the Work,
# where such license applies only to those patent claims licensable
# by such Contributor that are necessarily infringed by their
# Contribution(s) alone or by combination of their Contribution(s)
# with the Work to which such Contribution(s) was submitted. If You
# institute patent litigation against any entity (including a
# cross-claim or counterclaim in a lawsuit) alleging that the Work
# or a Contribution incorporated within the Work constitutes direct
# or contributory patent infringement, then any patent licenses
# granted to You under this License for that Work shall terminate
# as of the date such litigation is filed.
#
# 4. Redistribution. You may reproduce and distribute copies of the
# Work or Derivative Works thereof in any medium, with or without
# modifications, and in Source or Object form, provided that You
# meet the following conditions:
#
# (a) You must give any other recipients of the Work or
# Derivative Works a copy of this License; and
#
# (b) You must cause any modified files to carry prominent notices
# stating that You changed the files; and
#
# (c) You must retain, in the Source form of any Derivative Works
# that You distribute, all copyright, patent, trademark, and
# attribution notices from the Source form of the Work,
# excluding those notices that do not pertain to any part of
# the Derivative Works; and
#
# (d) If the Work includes a "NOTICE" text file as part of its
# distribution, then any Derivative Works that You distribute must
# include a readable copy of the attribution notices contained
# within such NOTICE file, excluding those notices that do not
# pertain to any part of the Derivative Works, in at least one
# of the following places: within a NOTICE text file distributed
# as part of the Derivative Works; within the Source form or
# documentation, if provided along with the Derivative Works; or,
# within a display generated by the Derivative Works, if and
# wherever such third-party notices normally appear. The contents
# of the NOTICE file are for informational purposes only and
# do not modify the License. You may add Your own attribution
# notices within Derivative Works that You distribute, alongside
# or as an addendum to the NOTICE text from the Work, provided
# that such additional attribution notices cannot be construed
# as modifying the License.
#
# You may add Your own copyright statement to Your modifications and
# may provide additional or different license terms and conditions
# for use, reproduction, or distribution of Your modifications, or
# for any such Derivative Works as a whole, provided Your use,
# reproduction, and distribution of the Work otherwise complies with
# the conditions stated in this License.
#
# 5. Submission of Contributions. Unless You explicitly state otherwise,
# any Contribution intentionally submitted for inclusion in the Work
# by You to the Licensor shall be under the terms and conditions of
# this License, without any additional terms or conditions.
# Notwithstanding the above, nothing herein shall supersede or modify
# the terms of any separate license agreement you may have executed
# with Licensor regarding such Contributions.
#
# 6. Trademarks. This License does not grant permission to use the trade
# names, trademarks, service marks, or product names of the Licensor,
# except as required for reasonable and customary use in describing the
# origin of the Work and reproducing the content of the NOTICE file.
#
# 7. Disclaimer of Warranty. Unless required by applicable law or
# agreed to in writing, Licensor provides the Work (and each
# Contributor provides its Contributions) on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied, including, without limitation, any warranties or conditions
# of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
# PARTICULAR PURPOSE. You are solely responsible for determining the
# appropriateness of using or redistributing the Work and assume any
# risks associated with Your exercise of permissions under this License.
#
# 8. Limitation of Liability. In no event and under no legal theory,
# whether in tort (including negligence), contract, or otherwise,
# unless required by applicable law (such as deliberate and grossly
# negligent acts) or agreed to in writing, shall any Contributor be
# liable to You for damages, including any direct, indirect, special,
# incidental, or consequential damages of any character arising as a
# result of this License or out of the use or inability to use the
# Work (including but not limited to damages for loss of goodwill,
# work stoppage, computer failure or malfunction, or any and all
# other commercial damages or losses), even if such Contributor
# has been advised of the possibility of such damages.
#
# 9. Accepting Warranty or Additional Liability. While redistributing
# the Work or Derivative Works thereof, You may choose to offer,
# and charge a fee for, acceptance of support, warranty, indemnity,
# or other liability obligations and/or rights consistent with this
# License. However, in accepting such obligations, You may act only
# on Your own behalf and on Your sole responsibility, not on behalf
# of any other Contributor, and only if You agree to indemnify,
# defend, and hold each Contributor harmless for any liability
# incurred by, or claims asserted against, such Contributor by reason
# of your accepting any such warranty or additional liability.
#
# END OF TERMS AND CONDITIONS
#
# APPENDIX: How to apply the Apache License to your work.
#
# To apply the Apache License to your work, attach the following
# boilerplate notice, with the fields enclosed by brackets "[]"
# replaced with your own identifying information. (Don't include
# the brackets!)  The text should be enclosed in the appropriate
# comment syntax for the file format. We also recommend that a
# file or class name and description of purpose be included on the
# same "printed page" as the copyright notice for easier
# identification within third-party archives.
#
# Copyright [yyyy] [name of copyright owner]
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Classes for training and, optionally, simultaneous validation.
"""

import copy
import datetime
import logging
import os
import time
import json
import re
from typing import List, Union
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import numpy as np
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.config import CfgNode
from detectron2.engine.defaults import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils import comm
from ginjinn.ginjinn_config import GinjinnConfiguration

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
    def build_train_loader(cls, cfg: CfgNode):
        """Build data loader for training.

        Parameters
        ----------
        cfg : CfgNode
            Detectron2 config.

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
    def from_ginjinn_config(cls, gj_cfg : GinjinnConfiguration) -> "Trainer":
        '''from_ginjinn_config

        Build Trainer object from GinjinnConfiguration instead of
        detectron2 configuration.

        Parameters
        ----------
        gj_cfg : GinjinnConfiguration
            GinjinnConfiguration object.

        Returns
        -------
        Trainer
            Trainer object
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

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(PlottingHook(self.cfg.TEST.EVAL_PERIOD, self.cfg.OUTPUT_DIR))
        return hooks


class ValTrainer(Trainer):
    """This trainer class evaluates validation data during training.
    """
    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str) -> COCOEvaluator:
        """Builds COCO evaluator for a given dataset.

        Parameters
        ----------
        cfg : CfgNode
            Detectron2 config.
        dataset_name : str
            Name of the evaluation data set.

        Returns
        ----------
        COCOEvaluator
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-2, LossEvalHook(
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
        hooks.append(PlottingHook(self.cfg.TEST.EVAL_PERIOD, self.cfg.OUTPUT_DIR))
        return hooks


def mapper_train(dataset_dict: dict, augmentations: list):
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
    # pylint: disable=E1101
    """
    This hook allows periodic loss calculation for the validation data set.
    It is executed every ``eval_period`` iterations and after the last iteration.

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
    def __init__(self, eval_period: int, model: torch.nn.Module, data_loader):
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
        self.trainer.storage.put_scalars(**losses_mean, smoothing_hint=False)

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


class PlottingHook(HookBase):
    """
    This hook provides periodic plotting of losses and evaluation scores.
    It is executed every ``eval_period`` iterations and after the last iteration.

    Parameters
    ----------
    eval_period : int
        Period to plot losses and evaluation scores. If set to 0, they are only calculated
        after the last iteration.
    outdir : str
        Output directory
    """
    def __init__(self, eval_period: int, outdir: str):
        self.period = eval_period
        self.outdir = outdir
        self.metrics_df = None
        self.pp = None

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self.period > 0 and next_iter % self.period == 0):
            self._plot_all()

    def _plot_all(self):
        """Plot all metrics logged in metrics.json into metrics.pdf.
        """
        json_file = os.path.join(self.outdir, "metrics.json")
        if not os.path.isfile(json_file):
            return

        entries = []
        with open(json_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                entries.append(entry)

        self.metrics_df = pd.DataFrame.from_records(entries).sort_values(by='iteration')
        colnames = self.metrics_df.columns.tolist()

        plt.ioff()
        self.pp = PdfPages(os.path.join(self.outdir, "metrics.pdf"))

        # plot losses
        p = re.compile(".*loss.*$")
        cols_sel = [s for s in colnames if p.match(s)]
        for col in cols_sel:
            colnames.remove(col)
        self._plot_losses(cols_sel, smooth=True)

        # plot evaluation scores (bbox)
        p = re.compile("^bbox/.*$")
        cols_sel = [s for s in colnames if p.match(s)]
        for col in cols_sel:
            colnames.remove(col)
        self._plot_metrics(
            cols_sel,
            nrow_grid = 2,
            ncol_grid = 3,
            dataset_name = "val",
            legend_pos = "lower right"
        )

        # plot evaluation scores (segmentation)
        p = re.compile("^segm/.*$")
        cols_sel = [s for s in colnames if p.match(s)]
        for col in cols_sel:
            colnames.remove(col)
        self._plot_metrics(
            cols_sel,
            nrow_grid = 2,
            ncol_grid = 3,
            dataset_name = "val",
            legend_pos = "lower right"
        )

        # plot remaining metrics
        self._plot_metrics(colnames, nrow_grid=3, ncol_grid=4)

        self.pp.close()

    def _plot_metrics(
        self,
        cols: List[str],
        nrow_grid: int = 2,
        ncol_grid: int = 3,
        width: Union[float, int] = 11.69,
        height: Union[float, int] = 8.27,
        dataset_name: str = None,
        legend_pos: str = "best"
    ):
        """Plot multiple metrics, arranged by a specfied grid.
        If the grid size is not sufficient to accomodate all subplots, additional pages
        are appended to the resulting PDF.

        Parameters
        ----------
        cols : list of str
            Names of data frame columns to be plotted
        nrow_grid : int
            Number of rows per PDF page
        ncol_grid : int
            Number of columns per PDF page
        width : float or int
            Page width in inches, defaults to A4 (landscape)
        height : float or int
            Page height in inches, defaults to A4 (landscape)
        dataset_name : str
            If specified, this is used as legend text.
        legend_pos : str
            Legend position, see matplotlib.pyplot.legend,
            e.g., "upper right", "lower left", "best", etc.
        """
        if not cols:
            return
        grid_size = nrow_grid * ncol_grid
        for i in range(0, len(cols), grid_size):
            cols_chunk = cols[i:i+grid_size]
            fig, axs = plt.subplots(nrow_grid, ncol_grid)
            fig.set_size_inches(width, height)
            for ax, metric_name in zip(axs.flat, cols_chunk):
                idcs = ~self.metrics_df[metric_name].isna()
                ax.plot(
                    self.metrics_df['iteration'][idcs],
                    self.metrics_df[metric_name][idcs],
                    label = dataset_name
                )
                ax.set_title(metric_name)
                if dataset_name:
                    ax.legend(loc=legend_pos)
            for ax in axs.flat:
                if not ax.lines:
                    ax.axis("off")
            fig.tight_layout()
            self.pp.savefig()

    def _plot_losses(
        self,
        cols: List[str],
        nrow_grid: int = 2,
        ncol_grid: int = 3,
        width: Union[float, int] = 11.69,
        height: Union[float, int] = 8.27,
        legend_pos: str = "best",
        smooth: bool = False,
        window_size: int = 10
    ):
        """Plot multiple losses, arranged by a specfied grid.
        In case a validation data set is used, its scores are combined with those of
        the training data set. If the grid size is not sufficient to accomodate all
        subplots, additional pages are appended to the resulting PDF.

        Parameters
        ----------
        cols : list of str
            Names of data frame columns (losses) to be plotted
        nrow_grid : int
            Number of rows per PDF page
        ncol_grid : int
            Number of columns per PDF page
        width : float or int
            Page width in inches, defaults to A4 (landscape)
        height : float or int
            Page height in inches, defaults to A4 (landscape)
        legend_pos : str
            Legend position, see matplotlib.pyplot.legend,
            e.g., "upper right", "lower left", "best", etc.
        smooth : bool
            If set to True, training losses are overlaid with their rolling mean.
        window_size : int
            Number of values to be averaged if smooth is set to True.
        """
        if not cols:
            return
        p = re.compile("^.*_val$")
        cols_train = [c for c in cols if not p.match(c)]
        grid_size = nrow_grid * ncol_grid
        for i in range(0, len(cols_train), grid_size):
            cols_chunk = cols_train[i:i+grid_size]
            fig, axs = plt.subplots(nrow_grid, ncol_grid)
            fig.set_size_inches(width, height)
            for ax, metric_name in zip(axs.flat, cols_chunk):
                idcs = ~self.metrics_df[metric_name].isna()
                if smooth:
                    ax.plot(
                        self.metrics_df['iteration'][idcs],
                        self.metrics_df[metric_name][idcs],
                        label="train",
                        color="tab:blue",
                        alpha=0.3
                    )
                    ax.plot(
                        self.metrics_df['iteration'][idcs],
                        self.metrics_df[metric_name][idcs].rolling(window_size).mean(),
                        label="train (smoothed)",
                        color="tab:blue"
                    )
                else:
                    ax.plot(
                        self.metrics_df['iteration'][idcs],
                        self.metrics_df[metric_name][idcs],
                        label="train",
                        color="tab:blue"
                    )
                if metric_name + "_val" in cols:
                    idcs = ~self.metrics_df[metric_name + "_val"].isna()
                    ax.plot(
                        self.metrics_df['iteration'][idcs],
                        self.metrics_df[metric_name + "_val"][idcs],
                        label="val",
                        color="tab:orange"
                    )
                ax.set_title(metric_name)
                ax.legend(loc=legend_pos, fontsize="small")
            for ax in axs.flat:
                if not ax.lines:
                    ax.axis("off")
            fig.tight_layout()
            self.pp.savefig()
