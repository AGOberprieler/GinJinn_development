'''
A module for mapping GinjinnAugmentationConfiguration to
Detectron2 compatible augmentations/transformations
'''

from detectron2.data import transforms as T
from .augmentation_error import InvalidAugmentationConfigurationTypeError
from ginjinn.ginjinn_config.augmentation_config import \
    GinjinnAugmentationConfiguration, \
    HorizontalFlipAugmentationConfiguration, \
    VerticalFlipAugmentationConfiguration

def map_horizontal_flip(aug_config: HorizontalFlipAugmentationConfiguration):
    '''Map HorizontalFlipAugmentationConfiguration to Detectron2 augmentation

    Parameters
    ----------
    aug_config : HorizontalFlipAugmentationConfiguration
        A HorizontalFlipAugmentationConfiguration object.

    Returns
    -------
    Augmentation
        Detectron2 augmentation object
    '''
    return T.RandomFlip(
        prob=aug_config.probability,
        horizontal=True,
        vertical=False
    )

def map_vertical_flip(aug_config: VerticalFlipAugmentationConfiguration):
    '''Map VerticalFlipAugmentationConfiguration to Detectron2 augmentation

    Parameters
    ----------
    aug_config : VerticalFlipAugmentationConfiguration
        A VerticalFlipAugmentationConfiguration object.

    Returns
    -------
    Augmentation
        Detectron2 augmentation object
    '''
    return T.RandomFlip(
        prob=aug_config.probability,
        horizontal=False,
        vertical=True
    )

AUGMENTATION_MAP = {
    HorizontalFlipAugmentationConfiguration: map_horizontal_flip,
    VerticalFlipAugmentationConfiguration: map_vertical_flip,
}

def map_ginjinn_augmentation(aug_config):
    '''Map a single ginjinn augmentation configuration to Detectron2 augmentation

    Parameters
    ----------
    aug_config
        A single GinJinn augmentation configuration,
        e.g. HorizontalFlipAugmentationConfiguration

    Returns
    -------
    Augmentation
        Detectron2 augmentation object

    Raises
    ------
    InvalidAugmentationConfigurationTypeError
        Raised if unknown augmentation configuration object is passed.
    '''
    mapping = None
    for aug_type, map_fun in AUGMENTATION_MAP.items():
        if isinstance(aug_config, aug_type):
            mapping = map_fun
            break

    if mapping is None:
        raise InvalidAugmentationConfigurationTypeError(
            'Unknown augmentation configuration type "{}".'.format(type(aug_config))
        )

    return mapping(aug_config)

def map_ginjinn_augmentation_configuration(
    aug_config: GinjinnAugmentationConfiguration
):
    '''Map GinjinnAugmentationConfiguration to a
    list of Detectron 2 augmentations.

    Parameters
    ----------
    aug_config : GinjinnAugmentationConfiguration
        A GinjinnAugmentationConfiguration object

    Returns
    -------
    Augmentations
        List of Detectron2 Augmentations
    '''

    augmentations = []
    for aug in aug_config.augmentations:
        augmentations.append(
            map_ginjinn_augmentation(aug)
        )

    return augmentations
