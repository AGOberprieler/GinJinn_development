'''
GinJinn augmentation configuration module
'''

# import copy
from typing import List
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

class HorizontalFlipAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Horizontal Flip Augmentation Configuration

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
        '''Build HorizontalFlipAugmentationConfiguration from dictionary

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

class VerticalFlipAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Vertical Flip Augmentation Configuration

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
        '''Build VerticalFlipAugmentation from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing vertical flip configurations.

        Returns
        -------
        VerticalFlipAugmentation
            VerticalFlipAugmentation object.
        '''
        probability = config.get('probability', 1.0)
        return cls(probability = probability)


class GinjinnAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn augmentation configurations.
    '''

    AVAILABLE_AUGMENTATIONS = {
        'horizontal_flip': HorizontalFlipAugmentationConfiguration,
        'vertical_flip': VerticalFlipAugmentationConfiguration,
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
    def from_dictionaries(cls, augmentation_dicts: List[dict]):
        '''Build augmentations configuration from list of dictionaries.

        Each augmentation dictionary should consist of single key naming
        the augmentation that should be performed with the a corresponding
        value, which is again a dictionary, listing the augmentation options.

        The following is an example for a horizontal flip augmentation dict:
        {
            'horizontal_flip': {
                probability: 0.25
            }
        }

        Parameters
        ----------
        augmentation_dicts : list[dict]
            List of dictionaries describing augmentations.

        Returns
        -------
        GinjinnAugmentationConfiguration
            GinjinnAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised when an invalid augmentation name is passed.
        '''

        augmentations = []
        for aug_dict in augmentation_dicts:
            # we expect only 1 key, see from_dictionaries documentation
            aug_name = list(aug_dict.keys())[0]
            aug_constructor = cls.AVAILABLE_AUGMENTATIONS.get(aug_name, None)
            if aug_constructor is None:
                raise InvalidAugmentationConfigurationError(
                    'Unknown augmentation "{}".'.format(aug_name)
                )

            aug = aug_constructor.from_dictionary(aug_dict[aug_name])
            augmentations.append(aug)

        return cls(augmentations)

    def _check_augmentations(self):
        '''Check augmentations for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised when an invalid augmentation was found.
        '''

        # nothing to check if there are no augmentations
        if len(self.augmentations) == 0:
            return

        for aug in self.augmentations:
            if not any(
                [isinstance(aug, av_aug) for av_aug in self.AVAILABLE_AUGMENTATIONS.values()]
            ):
                raise InvalidAugmentationConfigurationError(
                    'Unknown augmentation class "{}".'.format(type(aug))
                )
