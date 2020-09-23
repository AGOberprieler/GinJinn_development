'''
Detectron2 configuration module
'''

# import copy
# from typing import Optional

class GinjinnDetectronConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing additional Detectron2 configurations
    '''

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

        # TODO: implement
        return cls()
