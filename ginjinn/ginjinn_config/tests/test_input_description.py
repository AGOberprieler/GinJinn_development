'''Tests for GinjinnInputconfiguration
'''

import pytest
import copy
from ginjinn.ginjinn_config import GinjinnInputConfiguration, InvalidInputConfigurationError

@pytest.fixture
def basic_inputs():
    return [
        'PascalVOC',
        '/test/dir/annotations',
        '/test/dir/images'
    ]

@pytest.fixture
def custom_split_inputs():
    return [
        'test/dir/annotations_test',
        'test/dir/images_test',
        'test/dir/annotations_val',
        'test/dir/images_val'
    ]

@pytest.fixture
def automatic_split_inputs():
    return [
        0.2,
        0.2
    ]

@pytest.fixture
def config_dicts():
    simple_config = {
        'type': 'PascalVOC',
        'train': {
            'annotation_path': 'example_project_0/data/annotations',
            'image_path': 'example_project_0/data/images',
        },
    }
    test_custom_0 = {
        'test': {
            'annotation_path': 'test/dir/annotations_test',
            'image_path': 'test/dir/images_test',
        }
    }
    val_custom_0 = {
        'validation': {
            'annotation_path': 'test/dir/annotations_validation',
            'image_path': 'test/dir/images_validation',
        }
    }
    test_automatic_0 = {
        'split': {
            'test': 0.25
        }
    }
    val_automatic_0 = {
        'split': {
            'validation': 0.25
        }
    }

    test_custom_config_0 = copy.deepcopy(simple_config)
    test_custom_config_0.update(test_custom_0)

    val_custom_config_0 = copy.deepcopy(simple_config)
    val_custom_config_0.update(val_custom_0)

    test_val_custom_config_0 = copy.deepcopy(simple_config)
    test_val_custom_config_0.update(test_custom_0)
    test_val_custom_config_0.update(val_custom_0)

    test_val_automatic_config_0 = copy.deepcopy(simple_config)
    test_val_automatic_config_0.update(test_automatic_0)
    test_val_automatic_config_0['split']['validation'] = val_automatic_0['split']['validation']

    return [
        simple_config,
        test_custom_config_0,
        val_custom_config_0,
        test_val_custom_config_0,
        test_val_automatic_config_0,
    ]

def test_constructor_simple(basic_inputs):
    '''Simple constructor test.
    '''

    ann_type = basic_inputs[0]
    train_ann_path = basic_inputs[1]
    train_img_path = basic_inputs[2]

    input_configuration = GinjinnInputConfiguration(
        ann_type,
        train_ann_path,
        train_img_path,
    )

    assert input_configuration.type == ann_type,\
        'annotation type not set correctly'
    assert input_configuration.train['annotation_path'] == train_ann_path,\
        'train annotation path not set correctly'
    assert input_configuration.train['image_path'] == train_img_path,\
        'train image path not set correctly'
    
def test_constructor_custom_split(basic_inputs, custom_split_inputs):
    '''Test constructor with custom split.
    '''

    ann_type = basic_inputs[0]
    train_ann_path = basic_inputs[1]
    train_img_path = basic_inputs[2]

    test_ann_path = custom_split_inputs[0]
    test_img_path = custom_split_inputs[1]
    val_ann_path = custom_split_inputs[2]
    val_img_path = custom_split_inputs[3]

    # test split
    input_configuration_0 = GinjinnInputConfiguration(
        ann_type, train_ann_path, train_img_path,
        test_ann_path=test_ann_path,
        test_img_path=test_img_path
    )
    assert input_configuration_0.test['annotation_path'] == test_ann_path,\
        'test annotation path not set correctly'
    assert input_configuration_0.test['image_path'] == test_img_path,\
        'test image path not set correctly'


    # validation split
    input_configuration_1 = GinjinnInputConfiguration(
        ann_type, train_ann_path, train_img_path,
        val_ann_path=val_ann_path,
        val_img_path=val_img_path
    )
    assert input_configuration_1.validation['annotation_path'] == val_ann_path,\
        'validation annotation path not set correctly'
    assert input_configuration_1.validation['image_path'] == val_img_path,\
        'validation image path not set correctly'

    # test-validation split
    input_configuration_2 = GinjinnInputConfiguration(
        ann_type, train_ann_path, train_img_path,
        test_ann_path=test_ann_path,
        test_img_path=test_img_path,
        val_ann_path=val_ann_path,
        val_img_path=val_img_path
    )

    assert input_configuration_2.test['annotation_path'] == test_ann_path,\
        'test annotation path not set correctly (test-val)'
    assert input_configuration_2.test['image_path'] == test_img_path,\
        'test image path not set correctly (test-val)'
    assert input_configuration_2.validation['annotation_path'] == val_ann_path,\
        'validation annotation path not set correctly (test-val)'
    assert input_configuration_2.validation['image_path'] == val_img_path,\
        'validation image path not set correctly (test-val)'

def test_automatic_split(basic_inputs, automatic_split_inputs):
    ann_type = basic_inputs[0]
    train_ann_path = basic_inputs[1]
    train_img_path = basic_inputs[2]

    split_test=automatic_split_inputs[0]
    split_val=automatic_split_inputs[1]

    # test
    input_configuration_0 = GinjinnInputConfiguration(
        ann_type, train_ann_path, train_img_path,
        split_test=split_test
    )
    assert input_configuration_0.split['test'] == split_test,\
        'automatic test split not set correctly'

    # validation
    input_configuration_1 = GinjinnInputConfiguration(
        ann_type, train_ann_path, train_img_path,
        split_val=split_val
    )
    assert input_configuration_1.split['validation'] == split_val,\
        'automatic validation split not set correctly'

    # test-validation
    input_configuration_2 = GinjinnInputConfiguration(
        ann_type, train_ann_path, train_img_path,
        split_test=split_test,
        split_val=split_val
    )
    assert input_configuration_2.split['validation'] == split_val,\
        'automatic validation split not set correctly (test-val)'
    assert input_configuration_2.split['test'] == split_test,\
        'automatic test split not set correctly (test-val)'

def test_invalid_annotation_type(basic_inputs):
    ann_type = 'CVAT'
    train_ann_path = basic_inputs[1]
    train_img_path = basic_inputs[2]

    with pytest.raises(InvalidInputConfigurationError):
        input_configuration = GinjinnInputConfiguration(
            ann_type, train_ann_path, train_img_path
        )

def test_missing_test_val_paths(basic_inputs, custom_split_inputs):
    ann_type = basic_inputs[0]
    train_ann_path = basic_inputs[1]
    train_img_path = basic_inputs[2]

    test_ann_path = custom_split_inputs[0]
    test_img_path = custom_split_inputs[1]
    val_ann_path = custom_split_inputs[2]
    val_img_path = custom_split_inputs[3]

    with pytest.raises(InvalidInputConfigurationError):
        GinjinnInputConfiguration(
            ann_type, train_ann_path, train_img_path,
            test_ann_path=test_ann_path,
        )
    
    with pytest.raises(InvalidInputConfigurationError):
        GinjinnInputConfiguration(
            ann_type, train_ann_path, train_img_path,
            test_img_path=test_img_path,
        )
    
    with pytest.raises(InvalidInputConfigurationError):
        GinjinnInputConfiguration(
            ann_type, train_ann_path, train_img_path,
            val_ann_path=test_ann_path,
        )
    
    with pytest.raises(InvalidInputConfigurationError):
        GinjinnInputConfiguration(
            ann_type, train_ann_path, train_img_path,
            val_img_path=test_img_path,
        )

def test_contradictionary_inputs(basic_inputs, custom_split_inputs, automatic_split_inputs):
    ann_type = basic_inputs[0]
    train_ann_path = basic_inputs[1]
    train_img_path = basic_inputs[2]

    test_ann_path = custom_split_inputs[0]
    test_img_path = custom_split_inputs[1]
    val_ann_path = custom_split_inputs[2]
    val_img_path = custom_split_inputs[3]

    split_test=automatic_split_inputs[0]
    split_val=automatic_split_inputs[1]

    with pytest.raises(InvalidInputConfigurationError):
        GinjinnInputConfiguration(
            ann_type, train_ann_path, train_img_path,
            test_ann_path=test_ann_path,
            test_img_path=test_img_path,
            split_test=split_test,
        )
    
    with pytest.raises(InvalidInputConfigurationError):
        GinjinnInputConfiguration(
            ann_type, train_ann_path, train_img_path,
            val_ann_path=val_ann_path,
            val_img_path=val_img_path,
            split_test=split_test,
        )

    with pytest.raises(InvalidInputConfigurationError):
        GinjinnInputConfiguration(
            ann_type, train_ann_path, train_img_path,
            test_ann_path=test_ann_path,
            test_img_path=test_img_path,
            split_val=split_val,
        )

    with pytest.raises(InvalidInputConfigurationError):
        GinjinnInputConfiguration(
            ann_type, train_ann_path, train_img_path,
            val_ann_path=val_ann_path,
            val_img_path=val_img_path,
            split_val=split_val,
        )

def test_from_dictionary(config_dicts):
    simple_config = config_dicts[0]
    test_custom_config_0 = config_dicts[1]
    val_custom_config_0 = config_dicts[2]
    test_val_custom_config_0 = config_dicts[3]
    test_val_automatic_config_0 = config_dicts[4]

    input_configuration_0 = GinjinnInputConfiguration.from_dictionary(simple_config)
    assert input_configuration_0.type == simple_config['type'] and\
        input_configuration_0.train['annotation_path'] == simple_config['train']['annotation_path'] and\
        input_configuration_0.train['image_path'] == simple_config['train']['image_path'],\
        'Simple configuration from dictionary not successful.'

    input_configuration_1 = GinjinnInputConfiguration.from_dictionary(test_custom_config_0)
    assert input_configuration_1.type == test_custom_config_0['type'] and\
        input_configuration_1.train['annotation_path'] == test_custom_config_0['train']['annotation_path'] and\
        input_configuration_1.train['image_path'] == test_custom_config_0['train']['image_path'] and\
        input_configuration_1.test['annotation_path'] == test_custom_config_0['test']['annotation_path'] and\
        input_configuration_1.test['image_path'] == test_custom_config_0['test']['image_path'],\
        'Custom test configuration from dictionary not successful.'

    input_configuration_2 = GinjinnInputConfiguration.from_dictionary(val_custom_config_0)
    assert input_configuration_2.type == test_custom_config_0['type'] and\
        input_configuration_2.train['annotation_path'] == val_custom_config_0['train']['annotation_path'] and\
        input_configuration_2.train['image_path'] == val_custom_config_0['train']['image_path'] and\
        input_configuration_2.validation['annotation_path'] == val_custom_config_0['validation']['annotation_path'] and\
        input_configuration_2.validation['image_path'] == val_custom_config_0['validation']['image_path'],\
        'Custom validation configuration from dictionary not successful.'

    input_configuration_3 = GinjinnInputConfiguration.from_dictionary(test_val_custom_config_0)
    assert input_configuration_3.type == test_val_custom_config_0['type'] and\
        input_configuration_3.train['annotation_path'] == test_val_custom_config_0['train']['annotation_path'] and\
        input_configuration_3.train['image_path'] == test_val_custom_config_0['train']['image_path'] and\
        input_configuration_3.validation['annotation_path'] == test_val_custom_config_0['validation']['annotation_path'] and\
        input_configuration_3.validation['image_path'] == test_val_custom_config_0['validation']['image_path'] and\
        input_configuration_3.test['annotation_path'] == test_val_custom_config_0['test']['annotation_path'] and\
        input_configuration_3.test['image_path'] == test_val_custom_config_0['test']['image_path'],\
        'Custom test-validation configuration from dictionary not successful.'

    input_configuration_4 = GinjinnInputConfiguration.from_dictionary(test_val_automatic_config_0)
    assert input_configuration_4.type == test_val_automatic_config_0['type'] and\
        input_configuration_4.train['annotation_path'] == test_val_automatic_config_0['train']['annotation_path'] and\
        input_configuration_4.train['image_path'] == test_val_automatic_config_0['train']['image_path'] and\
        input_configuration_4.split['test'] == test_val_automatic_config_0['split']['test'] and\
        input_configuration_4.split['validation'] == test_val_automatic_config_0['split']['validation'],\
        'Automatic test-validation configuration from dictionary not successful.'
