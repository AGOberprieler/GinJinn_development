'''
GinJinn model configuration module
'''

import copy
from typing import Optional

class GinjinnModelConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn model configurations.
    '''
    def __init__( #pylint: disable=too-many-arguments
        self,
        name: str,
        learning_rate: float,
        batch_size: int,
    ):
        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size

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
        }

        # Maybe implement this more elegantly...
        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            name=config['name'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
        )
