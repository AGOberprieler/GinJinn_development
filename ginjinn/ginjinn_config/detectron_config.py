'''
Detectron2 configuration module
'''

# import copy
# from typing import Optional

class GinjinnDetectronConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing additional Detectron2 configurations

    Parameters
    ----------
    config : dict, optional
        A dictionary describing additional Detectron2 configurations, by default {}
    '''

    def __init__(self, config: dict = {}): #pylint: disable=dangerous-default-value
        self.config = config

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnAugmentationConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the augmentation configuration.

        Returns
        -------
        GinjinnDetectronConfiguration
            GinjinnDetectronConfiguration constructed with the configuration
            given in config.
        '''

        return cls(config)
