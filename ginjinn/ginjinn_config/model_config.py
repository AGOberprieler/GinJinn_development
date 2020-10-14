'''
GinJinn model configuration module
'''

import copy
# from typing import Optional
from .config_error import InvalidModelConfigurationError

# see all models: detectron2.model_zoo.model_zoo._ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX
MODEL_NAMES = [
    'faster_rcnn_R_50_C4_1x',
    'faster_rcnn_R_50_DC5_1x',
    'faster_rcnn_R_50_FPN_1x',
    'faster_rcnn_R_50_C4_3x',
    'faster_rcnn_R_50_DC5_3x',
    'faster_rcnn_R_50_FPN_3x',
    'faster_rcnn_R_101_C4_3x',
    'faster_rcnn_R_101_DC5_3x',
    'faster_rcnn_R_101_FPN_3x',
    'faster_rcnn_X_101_32x8d_FPN_3x',
    'mask_rcnn_R_50_C4_1x',
    'mask_rcnn_R_50_DC5_1x',
    'mask_rcnn_R_50_FPN_1x',
    'mask_rcnn_R_50_C4_3x',
    'mask_rcnn_R_50_DC5_3x',
    'mask_rcnn_R_50_FPN_3x',
    'mask_rcnn_R_101_C4_3x',
    'mask_rcnn_R_101_DC5_3x',
    'mask_rcnn_R_101_FPN_3x',
    'mask_rcnn_X_101_32x8d_FPN_3x'
]

class GinjinnModelConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn model configurations.

    Parameters
    ----------
    name : str
        model name/identifier.
    '''
    def __init__( #pylint: disable=too-many-arguments
        self,
        name: str,
    ):
        self.name = name

        if not self.name in MODEL_NAMES:
            raise InvalidModelConfigurationError('Invalid model name.')

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

        default_config = {}

        # Maybe implement this more elegantly...
        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            name=config['name'],
        )
