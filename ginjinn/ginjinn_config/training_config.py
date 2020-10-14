'''
GinJinn training configuration module
'''

import copy
# from typing import Optional
from .config_error import InvalidTrainingConfigurationError

class GinjinnTrainingConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn training configurations.

    Parameters
    ----------
    learning_rate : float
        learning rate for model training.
    batch_size : int
        batch size for model training and evaluation.
    max_iter: int
        maximum number of training iterations.
    '''
    def __init__( #pylint: disable=too-many-arguments
        self,
        learning_rate: float,
        batch_size: int,
        max_iter: int,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter

        self._check_config()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnTrainingConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the training configuration.

        Returns
        -------
        GinjinnTrainingConfiguration
            GinjinnTrainingConfiguration constructed with the configuration
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
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            max_iter=config['max_iter']
        )

    def _check_learning_rate(self):
        ''' Check learning rate config

        Raises
        ------
        InvalidTrainingConfigurationError
            Raised for invalid learning rate values.
        '''
        if self.learning_rate < 0:
            raise InvalidTrainingConfigurationError(
                'learning_rate must be greater than 0'
            )

    def _check_batch_size(self):
        ''' Check batch size config

        Raises
        ------
        InvalidTrainingConfigurationError
            Raised for invalid batch size values.
        '''
        if self.batch_size < 1:
            raise InvalidTrainingConfigurationError(
                'batch_size must be greater than or equal to 1'
            )

    def _check_max_iter(self):
        ''' Check max iter config

        Raises
        ------
        InvalidTrainingConfigurationError
            Raised for invalid max iter values.
        '''
        if self.max_iter < 1:
            raise InvalidTrainingConfigurationError(
                'max_iter must be greater than or equal to 1'
            )

    def _check_config(self):
        ''' Check configs
        '''
        self._check_learning_rate()
        self._check_batch_size()
        self._check_max_iter()
