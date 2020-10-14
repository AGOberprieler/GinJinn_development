'''
GinJinn augmentation configuration module
'''

import detectron2.data.transforms as T
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

    def __init__(self, probability: float = 1.0):
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

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomFlip(
            prob=self.probability,
            horizontal=True,
            vertical=False
        )

class VerticalFlipAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Vertical Flip Augmentation Configuration

    Parameters
    ----------
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(self, probability: float = 1.0):
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

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomFlip(
            prob=self.probability,
            horizontal=False,
            vertical=True
        )

class BrightnessAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''Random Brightness Augmentation Configuration

    Parameters
    ----------
    brightness_min : float
        Relative minimal brightness
    brightness_max : float
        Relative maximal brightness
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(
        self,
        brightness_min: float,
        brightness_max: float,
        probability: float = 1.0
    ):
        _check_probability(probability)

        self.probability = probability
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self._check_brightness()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build BrightnessAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing brightness configurations.

        Returns
        -------
        BrightnessAugmentationConfiguration
            BrightnessAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised when an invalid config is passed.
        '''
        probability = config.get('probability', 1.0)
        brightness_min = config.get('brightness_min', None)
        brightness_max = config.get('brightness_max', None)
        if brightness_min is None:
            raise InvalidAugmentationConfigurationError(
                '"brightness_min" required but not in config dictionary'
            )
        if brightness_max is None:
            raise InvalidAugmentationConfigurationError(
                '"brightness_max" required but not in config dictionary'
            )

        return cls(
            brightness_min=brightness_min,
            brightness_max=brightness_max,
            probability = probability
        )

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomApply(
            T.RandomBrightness(
                intensity_min=self.brightness_min,
                intensity_max=self.brightness_max,
            ),
            prob=self.probability
        )

    def _check_brightness(self):
        '''Check brightness values for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if brightness values not valid
        '''
        if self.brightness_min <= 0:
            raise InvalidAugmentationConfigurationError(
                'brightness_min must greather than 0.'
            )
        if self.brightness_max <= 0:
            raise InvalidAugmentationConfigurationError(
                'brightness_max must greather than 0.'
            )

        if self.brightness_min > self.brightness_max:
            raise InvalidAugmentationConfigurationError(
                'brightness_min must the less than brightness_max'
            )

class RotationRangeAugmentationConfiguration(): #pylint: disable=too-few-public-methods
    '''Rotation range augmentation

    Rotate randomly in the interval between angle_min and angle_max.

    Parameters
    ----------
    angle_min: float
        Minimum angle of rotation.
    angle_max: float
        Maximum angle of rotation.
    expand: bool
        image should be resized to fit the rotated image, alternatively cropped.
        By default True (resized).
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(
        self,
        angle_min: float,
        angle_max: float,
        expand: bool = True,
        probability: float = 1.0,
    ):
        _check_probability(probability)

        self.angle_min = angle_min
        self.angle_max = angle_max
        self.expand = expand
        self.probability = probability

        self._check_angles()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build RotationRangeAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing rotation configurations.

        Returns
        -------
        RotationRangeAugmentationConfiguration
            RotationRangeAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if required dictionary field is missing
        '''
        probability = config.get('probability', 1.0)
        expand = config.get('expand', True)
        angle_min = config.get('angle_min', None)
        angle_max = config.get('angle_max', None)

        if angle_min is None:
            raise InvalidAugmentationConfigurationError(
                '"angle_min" required but not in config dictionary'
            )
        if angle_max is None:
            raise InvalidAugmentationConfigurationError(
                '"angle_min" required but not in config dictionary'
            )

        return cls(
            angle_min=angle_min,
            angle_max=angle_max,
            expand=expand,
            probability = probability
        )

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomApply(
            T.RandomRotation(
                angle=(self.angle_min, self.angle_max),
                expand=self.expand,
                sample_style='range'
            ),
            prob=self.probability
        )

    def _check_angles(self):
        '''Check angles for validity

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if angles are not valid
        '''

        if self.angle_min > self.angle_max:
            raise InvalidAugmentationConfigurationError(
                'angle_min must the less than angle_max'
            )

class RotationChoiceAugmentationConfiguration(): #pylint: disable=too-few-public-methods
    '''Rotation selection augmentation

    Rotate randomly in the interval between angle_min and angle_max.

    Parameters
    ----------
    angles: list
        list of angles from which a random one will be chosen for each rotation augmentation.
    expand: bool
        image should be resized to fit the rotated image, alternatively cropped.
        By default True (resized).
    probability : float, optional
        Probability of applying the augmentation, by default 1.0 (always applied).
    '''

    def __init__(
        self,
        angles: list,
        expand: bool = True,
        probability: float = 1.0,
    ):
        _check_probability(probability)

        self.angles = angles
        self.expand = expand
        self.probability = probability

        self._check_angles()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build RotationChoiceAugmentationConfiguration from dictionary

        Parameters
        ----------
        config : dict
            Dictionary containing rotation configurations.

        Returns
        -------
        RotationChoiceAugmentationConfiguration
            RotationChoiceAugmentationConfiguration object.

        Raises
        ------
        InvalidAugmentationConfigurationError
            Raised if required dictionary field is missing
        '''
        probability = config.get('probability', 1.0)
        expand = config.get('expand', True)
        angles = config.get('angles', None)

        if angles is None:
            raise InvalidAugmentationConfigurationError(
                '"angles" required but not in config dictionary'
            )

        return cls(
            angles=angles,
            expand=expand,
            probability = probability
        )

    def to_detectron2_augmentation(self):
        '''Convert to Detectron2 augmentation

        Returns
        -------
        Augmentation
            Detectron2 augmentation
        '''
        return T.RandomApply(
            T.RandomRotation(
                angle=self.angles,
                expand=self.expand,
                sample_style='choice'
            ),
            prob=self.probability
        )

    def _check_angles(self):
        if len(self.angles) < 1:
            raise InvalidAugmentationConfigurationError(
                'There must be at least one angle to chose from.'
            )

class GinjinnAugmentationConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn augmentation configurations.
    '''

    AVAILABLE_AUGMENTATIONS = {
        'horizontal_flip': HorizontalFlipAugmentationConfiguration,
        'vertical_flip': VerticalFlipAugmentationConfiguration,
        'rotation_range': RotationRangeAugmentationConfiguration,
        'rotation_choice': RotationChoiceAugmentationConfiguration,
        'brightness': BrightnessAugmentationConfiguration,
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

    def to_detectron2_augmentations(self):
        '''Convert to Detectron2 augmentation list

        Returns
        -------
        Augmentations
            A list of Detectron2 augmentations
        '''
        augmentations = []
        for aug in self.augmentations:
            augmentations.append(aug.to_detectron2_augmentation())

        return augmentations

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
