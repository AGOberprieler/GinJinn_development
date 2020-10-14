''' Test module for augmentation_mapping.py
'''

import pytest

from ginjinn.augmentation import augmentation_mapping

def test_map_horizontal_flip():
    prob = 0.5
    hflip_config = augmentation_mapping.HorizontalFlipAugmentationConfiguration(
        prob
    )
    hflip_aug = augmentation_mapping.map_horizontal_flip(hflip_config)

    assert hflip_aug.prob == prob
    assert hflip_aug.horizontal == True
    assert hflip_aug.vertical == False

def test_map_vertical_flip():
    prob = 0.5
    vflip_config = augmentation_mapping.VerticalFlipAugmentationConfiguration(
        prob
    )
    hflip_aug = augmentation_mapping.map_vertical_flip(vflip_config)

    assert hflip_aug.prob == prob
    assert hflip_aug.horizontal == False
    assert hflip_aug.vertical == True

def test_invalid_augmentation_type():
    with pytest.raises(augmentation_mapping.InvalidAugmentationConfigurationTypeError):
        aug = augmentation_mapping.map_ginjinn_augmentation([1,2,3])
