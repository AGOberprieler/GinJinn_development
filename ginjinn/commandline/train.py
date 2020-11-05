''' Module for the ginjinn train subcommand.
'''

import os
from ginjinn.ginjinn_config import GinjinnConfiguration
import ginjinn.ginjinn_config.config_error as config_error

def ginjinn_train(args):
    '''ginjinn_train

    GinJinn train command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn train
        subcommand.
    '''
    project_dir = args.project_dir
    config_file = os.path.join(project_dir, 'ginjinn_config.yaml')

    if args.debug:
        config = GinjinnConfiguration.from_config_file(config_file)
    else:
        try:
            config = GinjinnConfiguration.from_config_file(config_file)
        except config_error.InvalidInputConfigurationError as iic_e:
            print('\nInvalid input configuration:')
            print(iic_e)
        #TODO: implement remaining exception checks.
        except Exception as any_e:
            raise any_e

    # TODO implement training
