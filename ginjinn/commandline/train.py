''' Module for the ginjinn train subcommand.
'''

import os
import sys
from ginjinn.ginjinn_config import GinjinnConfiguration
import ginjinn.ginjinn_config.config_error as config_error

def write_training(
    train_res: dict,
    file_path: str,
):
    '''write_training [summary]

    Parameters
    ----------
    train_res : dict
        [description]
    file_path : str
        [description]
    '''

def ginjinn_train(args):
    '''ginjinn_train

    GinJinn train command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn train
        subcommand.
    '''

    # import here to reduce startup time when train is not called.
    from ginjinn.data_reader.load_datasets import load_train_val_sets
    from ginjinn.trainer import ValTrainer, Trainer

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

    # register dataset(s) globally
    load_train_val_sets(config)

    # TODO implement training
    if config.input.val:
        trainer = ValTrainer.from_ginjinn_config(config)
    else:
        trainer = Trainer.from_ginjinn_config(config)

    resume = args.resume if not args.resume is None else config.options.resume

    trainer.resume_or_load(resume=resume)
    train_res = trainer.train()

    print(train_res)
