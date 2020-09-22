'''
A module for managing the representation of GinJinn configurations.
'''


# import copy
# from typing import Optional
import yaml
from .config_error import InvalidGinjinnConfigurationError
from .input_config import GinjinnInputConfiguration
from .model_config import GinjinnModelConfiguration
from .augmentation_config import GinjinnAugmentationConfiguration

TASKS = [
    'bbox-detection',
    # 'semantic-segmentation',
    'instance-segmentation',
]

class GinjinnConfiguration: #pylint: disable=too-many-arguments
    '''GinJinn configuration class.

    A class representing the configuration of a GinJinn project.

    Parameters
    ----------
    project_dir : str
        Project directory. All outputs will be written to this directory.
    task : str
        Object detection task type.
    input_configuration : GinjinnInputConfiguration
        Object describing the input.
    model_configuration : GinjinnModelConfiguration
        Object describing the model.
    augmentation_configuration : GinjinnAugmentationConfiguration
        Object describing the augmentation.

    Raises
    ------
    InvalidGinjinnConfigurationError
        If any of the general configuration is contradictionary or malformed.
    '''
    def __init__(
        self,
        project_dir: str,
        task: str,
        input_configuration: GinjinnInputConfiguration,
        model_configuration: GinjinnModelConfiguration,
        augmentation_configuration: GinjinnAugmentationConfiguration,
    ):
        self.project_dir = project_dir
        self.task = task
        self.input = input_configuration
        self.model = model_configuration
        self.augmentation = augmentation_configuration

        # task
        if not self.task in TASKS:
            raise InvalidGinjinnConfigurationError(
                '"task" must be one of {}'.format(TASKS)
            )

    # TODO: implement
    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnConfiguration from dictionary.

        Parameters
        ----------
        config : dict
            Dictionary object describing the GinJinn configuration.

        Returns
        -------
        GinjinnConfiguration
            GinjinnConfiguration constructed with the configuration
            given in config.
        '''

        input_configuration = GinjinnInputConfiguration.from_dictionary(
            config['input']
        )
        model_configuration = GinjinnModelConfiguration.from_dictionary(
            config['model']
        )
        augmentation_configuration = GinjinnAugmentationConfiguration.from_dictionary(
            config['augmentation']
        )

        return cls(
            project_dir=config['project_dir'],
            task=config['task'],
            input_configuration=input_configuration,
            model_configuration=model_configuration,
            augmentation_configuration=augmentation_configuration,
        )

    @classmethod
    def from_config_file(cls, file_path: str):
        '''Build GinjinnConfiguration from YAML configuration file.

        Parameters
        ----------
        file_path : str
            Path to GinJinn YAML configuration file.

        Returns
        -------
        GinjinnConfiguration
            GinjinnConfiguration constructed with the configuration
            given in the config file.
        '''

        with open(file_path) as config_file:
            config = yaml.safe_load(config_file)

        return cls.from_dictionary(config)
