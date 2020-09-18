'''
GinJinn input configuration module
'''

import copy
from typing import Optional
from .config_error import InvalidInputConfigurationError

ANNOTATION_TYPES = [
    'PascalVOC',
    'COCO'
]

class InputPaths: #pylint: disable=too-few-public-methods
    '''Class representing annotation and corresponding image paths.

    Parameters
    ----------
    ann_path : str
        Path to annotations. I.e. either a file or a folder path.
    img_path : str
        Path to the folder containing images.
        '''
    def __init__(
        self,
        ann_path: str,
        img_path: str,
    ):
        self.annotation_path = ann_path
        self.image_path = img_path

class SplitConfig: #pylint: disable=too-few-public-methods
    '''Class representing test and validation split options.

    Parameters
    ----------
    test_split : float
        Fraction of data set to use for testing.
    validation_split : float
        Fraction of data set to use for validation.
    '''
    def __init__(
        self,
        test_split: Optional[float] = None,
        validation_split: Optional[float] = None,
    ):
        self.test = test_split
        self.validation = validation_split

class GinjinnInputConfiguration: #pylint: disable=too-few-public-methods
    '''GinJinn input configuration class.

    A class representing the configuration of the input(s)
    for a GinJinn project. This includes the configuration
    or description of optionals train-validation-test
    splits of the data set.

    Train-validation-test can be
    - skipped, when leaving test_* and val_* arguments at default
    - custom, when specifying test_* and val_* arguments
    - automatic, when specifying split_* arguments

    Specifying test_*/val_* and split_* arguments at the same time is not allowed.

    Parameters
    ----------
    ann_type : str
        Type of the object detection annotations.
        "PascalVOC" or "COCO".
    train_ann_path : str
        Path to the directory containing annotations files for "PascalVOC".
        Path to the annotation file for "COCO".
    train_img_path : str
        Path to the directory containing the images.
    test_ann_path : Optional[str], optional
        Path to the directory containing annotations files for "PascalVOC".
        Path to the annotation file for "COCO".
    test_img_path : Optional[str], optional
        Path to the directory containing the images.
    val_ann_path : Optional[str], optional
        Path to the directory containing annotations files for "PascalVOC".
        Path to the annotation file for "COCO".
    val_img_path : Optional[str], optional
        Path to the directory containing the images.
    split_test : Optional[float], optional
        Fraction of the dataset to use for testing.
    split_val : Optional[float], optional
        Fraction of the dataset to use for validation.

    Raises
    ------
    InvalidInputConfigurationError
        If the input configuration is contradictionary or malformed.
    '''

    def __init__( #pylint: disable=too-many-arguments
        self,
        ann_type: str,
        train_ann_path: str,
        train_img_path: str,
        test_ann_path: Optional[str] = None,
        test_img_path: Optional[str] = None,
        val_ann_path: Optional[str] = None,
        val_img_path: Optional[str] = None,
        split_test: Optional[float] = None,
        split_val: Optional[float] = None,
    ):
        self.type = ann_type
        # self.train = {
        #     'annotation_path': train_ann_path,
        #     'image_path': train_img_path,
        # }
        self.train = InputPaths(train_ann_path, train_img_path)

        self.test = None
        self.validation = None
        self.split = None

        # type
        if not self.type in ANNOTATION_TYPES:
            raise InvalidInputConfigurationError(
                '"ann_type" must be one of {}.'.format(ANNOTATION_TYPES)
            )

        # test
        if (not test_ann_path is None) or (not test_img_path is None):
            if (test_ann_path is None) or (test_img_path is None):
                raise InvalidInputConfigurationError(
                    'If any of "test_ann_path" and "test_img_path" is passed, \
                    the other must be passed too.'
                )
            # self.test = {
            #     'annotation_path': test_ann_path,
            #     'image_path': test_img_path,
            # }
            self.test = InputPaths(test_ann_path, test_img_path)

        # test
        if (not val_ann_path is None) or (not val_img_path is None):
            if (val_ann_path is None) or (val_img_path is None):
                raise InvalidInputConfigurationError(
                    'If any of "val_ann_path" and "val_img_path" is passed, \
                    the other must be passed too.'
                )
            # self.validation = {
            #     'annotation_path': val_ann_path,
            #     'image_path': val_img_path,
            # }
            self.validation = InputPaths(val_ann_path, val_img_path)

        # split
        if (not split_test is None) or (not split_val is None):
            # self.split = {}
            self.split = SplitConfig()
            if not split_test is None:
                # self.split['test'] = split_test
                self.split.test = split_test
            if not split_val is None:
                # self.split['validation'] = split_val
                self.split.validation = split_val

        # check whether contradicting parameters were passed
        is_custom = (not self.test is None) or (not self.validation is None)
        is_automatic = not self.split is None
        if is_custom and is_automatic:
            raise InvalidInputConfigurationError(
                'Specifying "test_*/val_*" and "split_*" arguments at the same time is not \
                allowed. Either pass "test_*/val_*" arguments for a custom train-validation-test \
                split or specify options for automatic split via "split_*" arguments.'
            )

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnInputConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the input configuration.

        Returns
        -------
        GinjinnInputConfiguration
            GinjinnInputConfiguration constructed with the configuration
            given in config.
        '''

        default_config = {
            'test': {
                'annotation_path': None,
                'image_path': None,
            },
            'validation': {
                'annotation_path': None,
                'image_path': None,
            },
            'split': {
                'test': None,
                'validation': None
            }
        }

        # Maybe implement this more elegantly...
        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            ann_type = config['type'],
            train_ann_path = config['train']['annotation_path'],
            train_img_path = config['train']['image_path'],
            test_ann_path = config['test']['annotation_path'],
            test_img_path = config['test']['image_path'],
            val_ann_path = config['validation']['annotation_path'],
            val_img_path = config['validation']['image_path'],
            split_test = config['split']['test'],
            split_val = config['split']['validation'],
        )
