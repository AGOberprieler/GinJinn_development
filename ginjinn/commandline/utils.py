''' Module for the ginjinn utils subcommand.
'''

import os
import shutil
import sys

from ginjinn.utils import confirmation_cancel
from ginjinn.utils import flatten_coco

def ginjinn_utils(args):
    '''ginjinn_utils

    GinJinn utils command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        subcommand.
    '''

    if args.utils_subcommand == 'merge':
        utils_merge(args)
    if args.utils_subcommand == 'cleanup':
        utils_cleanup(args)
    if args.utils_subcommand == 'flatten':
        utils_flatten(args)
    else:
        err = f'Unknown utils subcommand "{args.utils_subcommand}".'
        raise Exception(err)

def utils_merge(args):
    '''utils_merge

    GinJinn utils merge command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils merge
        subcommand.
    '''

    print(args)

    image_dirs = [x[0] for x in args.image_dir]
    ann_paths = [x[0] for x in args.ann_path]

    print(image_dirs)
    print(ann_paths)
    # TODO implement!

def utils_cleanup(args):
    '''utils_cleanup

    GinJinn utils cleanup command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils cleanup
        subcommand.
    '''
    project_dir = args.project_dir

    eval_res_path = os.path.join(project_dir, 'evaluation.csv')
    if os.path.exists(eval_res_path):
        os.remove(eval_res_path)
        print(f'Removed "{eval_res_path}" ...')

    class_names_path = os.path.join(project_dir, 'class_names.txt')
    if os.path.exists(class_names_path):
        os.remove(class_names_path)
        print(f'Removed "{class_names_path}" ...')

    outputs_path = os.path.join(project_dir, 'outputs')
    if os.path.isdir(outputs_path):
        shutil.rmtree(outputs_path)
        os.mkdir(outputs_path)
        print(f'Cleaned up "{outputs_path}" ...')

    print(f'Project "{project_dir}" cleaned up.')

def utils_flatten(args):
    '''utils_clutils_flatteneanup

    GinJinn utils flatten command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils flatten
        subcommand.
    '''

    out_dir = args.out_dir
    image_root_dir = args.image_root_dir
    ann_path = args.ann_path
    sep = args.seperator
    custom_id = args.custom_id

    if os.path.exists(out_dir):
        if confirmation_cancel(
            f'\nDirectory "{out_dir}" already exists.\nDo you want to overwrite it? ' + \
            f'WARNING: this will delete "{out_dir}" and ALL SUBDIRECTORIES!\n'
        ):
            shutil.rmtree(out_dir)
        else:
            sys.exit()

    os.mkdir(out_dir)

    flatten_coco(
        ann_file=ann_path,
        img_root_dir=image_root_dir,
        out_dir=out_dir,
        sep=sep,
        custom_id=custom_id,
    )

    print(f'Flattened data set written to {out_dir}.')
