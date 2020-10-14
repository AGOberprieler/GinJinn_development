''' Module to test augmentation_config.py
'''

import pytest

from ginjinn.ginjinn_config import GinjinnAugmentationConfiguration, InvalidAugmentationConfigurationError
from ginjinn.ginjinn_config.augmentation_config import HorizontalFlipAugmentationConfiguration, \
    VerticalFlipAugmentationConfiguration, \
    RotationRangeAugmentationConfiguration, \
    RotationChoiceAugmentationConfiguration

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
            },
            {
                'rotation_range': {
                    'angle_min': -10,
                    'angle_max': 10,
                    'expand': True,
                    'probability': 0.25
                }
            },
            {
                'rotation_choice': {
                    'angles': [
                        -10,
                        -20,
                        10,
                        20,
                    ],
                    'expand': True,
                    'probability': 0.25
                }
            }
        ],
        [
            HorizontalFlipAugmentationConfiguration,
            VerticalFlipAugmentationConfiguration,
            RotationRangeAugmentationConfiguration,
            RotationChoiceAugmentationConfiguration,
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

    assert len(aug.augmentations) == len(simple_augmentation_list[0])
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

def test_invalid_rotation_range():
    with pytest.raises(InvalidAugmentationConfigurationError):
        RotationRangeAugmentationConfiguration.from_dictionary(
            {
                'angle_min': 11,
                'angle_max': 10,
                'expand': True,
                'probability': 0.25
            }
        )
    with pytest.raises(InvalidAugmentationConfigurationError):
        RotationRangeAugmentationConfiguration.from_dictionary(
            {
                'angle_min': -10,
                'expand': True,
                'probability': 0.25
            }
        )
    with pytest.raises(InvalidAugmentationConfigurationError):
        RotationRangeAugmentationConfiguration.from_dictionary(
            {
                'angle_max': 20,
                'expand': True,
                'probability': 0.25
            }
        )

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

    assert d_augs[2].prob == simple_augmentation_list[0][2]['rotation_range']['probability']
    assert d_augs[2].transform.angle[0] == simple_augmentation_list[0][2]['rotation_range']['angle_min']
    assert d_augs[2].transform.angle[1] == simple_augmentation_list[0][2]['rotation_range']['angle_max']
    assert d_augs[2].transform.expand == simple_augmentation_list[0][2]['rotation_range']['expand']

    assert d_augs[3].prob == simple_augmentation_list[0][3]['rotation_choice']['probability']
    l1 = len(d_augs[3].transform.angle)
    l2 = len(simple_augmentation_list[0][3]['rotation_choice']['angles'])
    assert l1 == l2
    for a1, a2 in zip(
        d_augs[3].transform.angle,
        simple_augmentation_list[0][3]['rotation_choice']['angles']
    ):
        assert a1 == a2
    
    assert d_augs[3].transform.expand == simple_augmentation_list[0][3]['rotation_choice']['expand']
