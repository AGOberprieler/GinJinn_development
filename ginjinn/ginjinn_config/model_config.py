'''
GinJinn model configuration module
'''

import copy
import os
# from typing import Optional
from .config_error import InvalidModelConfigurationError

# see all models: detectron2.model_zoo.model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX
MODEL_NAMES = {
    'faster_rcnn_R_50_C4_1x': 'COCO-Detection/faster_rcnn_R_50_C4_1x.yaml',
    'faster_rcnn_R_50_DC5_1x': 'COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml',
    'faster_rcnn_R_50_FPN_1x': 'COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml',
    'faster_rcnn_R_50_C4_3x': 'COCO-Detection/faster_rcnn_R_50_C4_3x.yaml',
    'faster_rcnn_R_50_DC5_3x': 'COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml',
    'faster_rcnn_R_50_FPN_3x': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
    'faster_rcnn_R_101_C4_3x': 'COCO-Detection/faster_rcnn_R_101_C4_3x.yaml',
    'faster_rcnn_R_101_DC5_3x': 'COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml',
    'faster_rcnn_R_101_FPN_3x': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
    'faster_rcnn_X_101_32x8d_FPN_3x': 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
    'mask_rcnn_R_50_C4_1x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml',
    'mask_rcnn_R_50_DC5_1x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml',
    'mask_rcnn_R_50_FPN_1x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
    'mask_rcnn_R_50_C4_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml',
    'mask_rcnn_R_50_DC5_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml',
    'mask_rcnn_R_50_FPN_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    'mask_rcnn_R_101_C4_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml',
    'mask_rcnn_R_101_DC5_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml',
    'mask_rcnn_R_101_FPN_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
    'mask_rcnn_X_101_32x8d_FPN_3x': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
}

MODEL_TASKS = {
    'faster_rcnn_R_50_C4_1x': ['bbox-detection'],
    'faster_rcnn_R_50_DC5_1x': ['bbox-detection'],
    'faster_rcnn_R_50_FPN_1x': ['bbox-detection'],
    'faster_rcnn_R_50_C4_3x': ['bbox-detection'],
    'faster_rcnn_R_50_DC5_3x': ['bbox-detection'],
    'faster_rcnn_R_50_FPN_3x': ['bbox-detection'],
    'faster_rcnn_R_101_C4_3x': ['bbox-detection'],
    'faster_rcnn_R_101_DC5_3x': ['bbox-detection'],
    'faster_rcnn_R_101_FPN_3x': ['bbox-detection'],
    'faster_rcnn_X_101_32x8d_FPN_3x': ['bbox-detection'],
    'mask_rcnn_R_50_C4_1x': ['instance-segmentation'],
    'mask_rcnn_R_50_DC5_1x': ['instance-segmentation'],
    'mask_rcnn_R_50_FPN_1x': ['instance-segmentation'],
    'mask_rcnn_R_50_C4_3x': ['instance-segmentation'],
    'mask_rcnn_R_50_DC5_3x': ['instance-segmentation'],
    'mask_rcnn_R_50_FPN_3x': ['instance-segmentation'],
    'mask_rcnn_R_101_C4_3x': ['instance-segmentation'],
    'mask_rcnn_R_101_DC5_3x': ['instance-segmentation'],
    'mask_rcnn_R_101_FPN_3x': ['instance-segmentation'],
    'mask_rcnn_X_101_32x8d_FPN_3x': ['instance-segmentation'],
}

# TODO: implement model-specific parameters

class GinjinnModelConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn model configurations.

    Parameters
    ----------
    name : str
        model name/identifier.
    initial_weights : str
        Determines the initialization of the model weights.
        One of
            - 'random', meaning random weights initialization
            - 'pretrained', meaning pretrained weights from the Detectron2 model zoo, if available
        or the file path of a weights file.

    Raises
    ------
    InvalidModelConfigurationError
        Raised if invalid model name is passed.
    '''
    def __init__( #pylint: disable=too-many-arguments
        self,
        name: str,
        initial_weights: str,
    ):
        self.name = name
        if not name in MODEL_NAMES.keys():
            raise InvalidModelConfigurationError('Invalid model name.')

        self.detectron2_config_name = MODEL_NAMES[self.name]

        self.initial_weights = initial_weights
        self._check_initial_weights()

    def to_detectron2_config(self):
        '''to_detectron2_config

        Convert model configuration to Detectron2 configuration.

        Returns
        -------
        detectron2_config
            Detectron2 configuration.
        '''

        # import here to reduce loading times, when detectron2 conversion is not
        # required
        from detectron2.config import get_cfg #pylint: disable=import-outside-toplevel
        from detectron2.model_zoo import get_config_file, get_checkpoint_url #pylint: disable=import-outside-toplevel

        cfg = get_cfg()
        model_config_file = get_config_file(self.detectron2_config_name)
        cfg.merge_from_file(model_config_file)

        if self.initial_weights == 'pretrained':
            model_url = get_checkpoint_url(self.detectron2_config_name)
            cfg.MODEL.WEIGHTS = model_url
        else:
            cfg.MODEL.WEIGHTS = ''

        # TODO:
        # model_parameters

        return cfg

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnModelConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the model configuration.

        Returns
        -------
        GinjinnModelConfiguration
            GinjinnModelConfiguration constructed with the configuration
            given in config.
        '''

        default_config = {
            'initial_weights': 'random'
        }

        # Maybe implement this more elegantly...
        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            name=config['name'],
            initial_weights=config['initial_weights'],
        )

    def _check_initial_weights(self):
        '''Check initial_weights option

        Raises
        ------
        InvalidModelConfigurationError
            Raised if an invalid initial_weights option is passed.
        '''

        if self.initial_weights != 'random' and self.initial_weights != 'pretrained':
            print(self.initial_weights)
            if not os.path.isfile(self.initial_weights):
                raise InvalidModelConfigurationError(
                    'initial_weights must be either "random", "pretrained", or a valid weights file path.' #pylint: disable=line-too-long
                )
