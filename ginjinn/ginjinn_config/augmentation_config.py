'''
GinJinn augmentation configuration module
'''

# import copy
# from typing import Optional
from .config_error import InvalidAugmentationConfigurationError

def _check_probability(probability: float):
    '''Helper function to check augmentation probabilities

    Parameters
    ----------
    probability : float
        Augmentation probability

    Raises
    ------
    InvalidAugmentationConfigurationError
        Raised when an invalid probability value is passed.
    '''
    if probability < 0.0 or probability > 1.0:
        raise InvalidAugmentationConfigurationError(
            'The probability of an augmentation must be between 0.0 and 1.0.'
        )

class HorizontalFlipAugmentation:
    '''Horizontal Flip Augmentation

    Parameters
    ----------
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(self, probability: float):
        _check_probability(probability)

        self.probability = probability

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build HorizontalFlipAugmentation from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing horizontal flip configurations.

        Returns
        -------
        HorizontalFlipAugmentation
            HorizontalFlipAugmentation object.
        '''
        probability = config.get('probability', 1.0)
        return cls(probability = probability)



class GinjinnAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn augmentation configurations.
    '''

    AVAILABLE_AUGMENTATIONS = {
        'horizontal_flip': HorizontalFlipAugmentation,
    }

    def __init__(
        self,
        augmentations: list
    ):
        '''Class representing augmentation configurations

        Parameters
        ----------
        augmentations : list
            List of Augmentation objects.
        '''
        self.augmentations = augmentations
        self._check_augmentations()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnAugmentationConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the augmentation configuration.

        Returns
        -------
        GinjinnAugmentationConfiguration
            GinjinnAugmentationConfiguration constructed with the configuration
            given in config.
        '''

        # TODO: implement
        return cls()

    def _check_augmentations(self):
        '''Check augmentations for validity
        '''

        # TODO
        pass
