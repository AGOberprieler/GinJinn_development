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
    learning_rate : float
        learning rate for model training.
    batch_size : int
        batch size for model training and evaluation.
    max_iter: int
        maximum number of training iterations.
    '''
    def __init__( #pylint: disable=too-many-arguments
        self,
        name: str,
        learning_rate: float,
        batch_size: int,
        max_iter: int,
    ):
        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter


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

        default_config = {
            'learning_rate': 0.002,
            'batch_size': 1,
            'max_iter': 40000,
        }

        # Maybe implement this more elegantly...
        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            name=config['name'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            max_iter=config['max_iter']
        )

    def _check_learning_rate(self):
        ''' Check learning rate config

        Raises
        ------
        InvalidModelConfigurationError
            Raised for invalid learning rate values.
        '''
        if self.learning_rate < 0:
            raise InvalidModelConfigurationError('learning_rate must be greater than 0')

    def _check_batch_size(self):
        ''' Check batch size config

        Raises
        ------
        InvalidModelConfigurationError
            Raised for invalid batch size values.
        '''
        if self.batch_size < 1:
            raise InvalidModelConfigurationError('batch_size must be greater than or equal to 1')

    def _check_max_iter(self):
        ''' Check max iter config

        Raises
        ------
        InvalidModelConfigurationError
            Raised for invalid max iter values.
        '''
        if self.max_iter < 1:
            raise InvalidModelConfigurationError('max_iter must be greater than or equal to 1')

    def _check_config(self):
        ''' Check configs
        '''
        self._check_learning_rate()
        self._check_batch_size()
        self._check_max_iter()
