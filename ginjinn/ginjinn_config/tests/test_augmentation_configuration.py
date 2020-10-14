''' Module to test augmentation_config.py
'''

import pytest

from ginjinn.ginjinn_config import GinjinnAugmentationConfiguration, InvalidAugmentationConfigurationError
from ginjinn.ginjinn_config.augmentation_config import HorizontalFlipAugmentationConfiguration, VerticalFlipAugmentationConfiguration

@pytest.fixture
def simple_augmentation_list():
    return (
        [
            {
                'horizontal_flip': {
                    'probability': 0.25
                }
            },
            {
                'vertical_flip': {
                    'probability': 0.25
                }
            }
        ],
        [
            HorizontalFlipAugmentationConfiguration,
            VerticalFlipAugmentationConfiguration
        ]
    )

@pytest.fixture
def invalid_augmentation_list():
    return [
        {
            'invalid_augmentation': {
                'probability': 0.25
            }
        },
    ]

def test_simple(simple_augmentation_list):
    aug = GinjinnAugmentationConfiguration.from_dictionaries(
        simple_augmentation_list[0]
    )

    assert len(aug.augmentations) == len(simple_augmentation_list)
    assert isinstance(aug.augmentations[0], simple_augmentation_list[1][0])
    assert aug.augmentations[0].probability == simple_augmentation_list[0][0]['horizontal_flip']['probability']
    assert isinstance(aug.augmentations[1], simple_augmentation_list[1][1])
    assert aug.augmentations[1].probability == simple_augmentation_list[0][1]['vertical_flip']['probability']

def test_invalid_aug_name(invalid_augmentation_list):
    with pytest.raises(InvalidAugmentationConfigurationError):
        aug = GinjinnAugmentationConfiguration.from_dictionaries(invalid_augmentation_list)

def test_invalid_aug_class():
    with pytest.raises(InvalidAugmentationConfigurationError):
        aug = GinjinnAugmentationConfiguration([{}, {}])

def test_empty_aug():
    aug_1 = GinjinnAugmentationConfiguration.from_dictionaries([])
    assert len(aug_1.augmentations) == 0

    aug_2 = GinjinnAugmentationConfiguration([])
    assert len(aug_2.augmentations) == 0

def test_invalid_probability():
    with pytest.raises(InvalidAugmentationConfigurationError):
        aug = GinjinnAugmentationConfiguration.from_dictionaries([
            {'horizontal_flip': {
                'probability': -0.1
            }}
        ])

    with pytest.raises(InvalidAugmentationConfigurationError):
        aug = GinjinnAugmentationConfiguration.from_dictionaries([
            {'horizontal_flip': {
                'probability': 1.1
            }}
        ])

def test_detectron2_conversion(simple_augmentation_list):
    aug = GinjinnAugmentationConfiguration.from_dictionaries(
        simple_augmentation_list[0]
    )

    d_augs = aug.to_detectron2_augmentations()

    assert d_augs[0].prob == simple_augmentation_list[0][0]['horizontal_flip']['probability']
    assert d_augs[0].horizontal == True
    assert d_augs[0].vertical == False

    assert d_augs[1].prob == simple_augmentation_list[0][1]['vertical_flip']['probability']
    assert d_augs[1].horizontal == False
    assert d_augs[1].vertical == True
