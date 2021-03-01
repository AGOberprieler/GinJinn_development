''' Module for the ginjinn new subcommand
'''

import shutil
import sys
import os

import pkg_resources

from ginjinn.utils import confirmation_cancel

def ginjinn_new(args):
    '''ginjinn_new

    GinJinn new command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn new
        subcommand.
    '''

    template_dir = pkg_resources.resource_filename(
        'ginjinn', 'data/ginjinn_config/templates',
    )
    template_path = os.path.join(template_dir, args.template)


    project_dir = args.project_dir
    if os.path.exists(project_dir):
        if confirmation_cancel(
            f'\nDirectory "{project_dir}" already exists.\nDo you want to overwrite it? ' + \
            f'WARNING: this will delete "{project_dir}" and ALL SUBDIRECTORIES!\n'
        ):
            shutil.rmtree(project_dir)
        else:
            sys.exit()

    os.mkdir(project_dir)
    os.mkdir(os.path.join(project_dir, 'outputs'))

    with open(template_path) as cfg_template_file:
        config_str = cfg_template_file.read()
    config_str = config_str.replace('"ENTER PROJECT DIR HERE"', f'{os.path.abspath(project_dir)}')

    config_path = os.path.join(project_dir, 'ginjinn_config.yaml')
    with open(config_path, 'w') as cfg_file:
        cfg_file.write(config_str)

    print(f'Initialized GinJinn project at "{project_dir}".')
