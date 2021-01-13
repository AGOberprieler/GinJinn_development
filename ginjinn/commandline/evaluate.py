''' Module for the ginjinn evaluate subcommand.
'''

import os
import sys
from ginjinn.ginjinn_config import GinjinnConfiguration
import ginjinn.ginjinn_config.config_error as config_error

def write_evaluation(eval_res: dict):
    '''write_evaluation

    Write evaluation results.

    Parameters
    ----------
    eval_res : dict
        Dictionary containing the evalaution results
    '''

def ginjinn_evaluate(args):
    '''ginjinn_evaluate

    GinJinn evaluate command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn evaluate
        subcommand.
    '''

    project_dir = args.project_dir
    config_file = os.path.join(project_dir, 'ginjinn_config.yaml')

    if args.debug:
        config = GinjinnConfiguration.from_config_file(config_file)
    else:
        try:
            config = GinjinnConfiguration.from_config_file(config_file)
        except config_error.InvalidInputConfigurationError as err:
            print('\nInvalid input configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidModelConfigurationError as err:
            print('\nInvalid model configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidAugmentationConfigurationError as err:
            print('\nInvalid augmentation configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidGinjinnConfigurationError as err:
            print('\nInvalid GinJinn configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidOptionsConfigurationError as err:
            print('\nInvalid options configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidTrainingConfigurationError as err:
            print('\nInvalid training configuration:')
            print(err)
            sys.exit(1)
        except Exception as any_e:
            raise any_e

    # import here to reduce startup time when train is not called.
    from ginjinn.evaluation import evaluate
    from ginjinn.data_reader.load_datasets import load_test_set

    # register data set globally
    load_test_set(config)

    res = evaluate(config)
    print(res)
