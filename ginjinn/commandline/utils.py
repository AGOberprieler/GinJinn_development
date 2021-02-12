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
    elif args.utils_subcommand == 'cleanup':
        utils_cleanup(args)
    elif args.utils_subcommand == 'flatten':
        utils_flatten(args)
    elif args.utils_subcommand == 'crop':
        utils_crop(args)
    elif args.utils_subcommand == 'sliding_window':
        utils_sliding_window(args)
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

    from ginjinn.data_reader.merge_datasets import merge_datasets_coco, merge_datasets_pvoc

    image_dirs = [x[0] for x in args.image_dir]
    ann_paths = [x[0] for x in args.ann_path]

    out_dir = args.out_dir
    ann_type = args.ann_type

    link_images = args.link_images

    if ann_type == 'COCO':
        merge_datasets_coco(
            ann_files=ann_paths,
            img_dirs=image_dirs,
            outdir=out_dir,
            link_images=link_images,
        )
    elif ann_type == 'PVOC':
        merge_datasets_pvoc(
            ann_dirs=ann_paths,
            img_dirs=image_dirs,
            outdir=out_dir,
            link_images=link_images,
        )

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
    '''utils_flatten

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
    annotated_only = args.annotated_only

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
        annotated_only=annotated_only,
    )

    print(f'Flattened data set written to {out_dir}.')

def utils_crop(args):
    '''utils_crop

    GinJinn utils crop command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils crop
        subcommand.
    '''

    from ginjinn.utils import crop_seg_from_coco

    crop_seg_from_coco(
        ann_file=args.ann_path,
        img_dir=args.image_dir,
        outdir=args.out_dir,
        padding=args.padding,
    )

def utils_sliding_window(args):
    '''utils_crop

    GinJinn utils sliding_window command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn utils
        sliding_window subcommand.
    '''

    from ginjinn.utils.dataset_cropping import sliding_window_crop_coco, sliding_window_crop_pvoc

    if os.path.exists(args.out_dir):
        msg = f'Directory "{args.out_dir} already exists. Should it be overwritten?"\n' +\
            f'WARNING: This will remove "{args.out_dir}" and ALL SUBDIRECTORIES.\n'
        should_remove = confirmation_cancel(msg)
        if should_remove:
            shutil.rmtree(args.out_dir)
            os.mkdir(args.out_dir)
    else:
        os.mkdir(args.out_dir)

    img_dir_out = os.path.join(args.out_dir, 'images')
    if os.path.exists(img_dir_out):
        msg = f'Directory "{img_dir_out} already exists. Should it be overwritten?"\n' +\
            f'WARNING: This will remove "{img_dir_out}" and ALL SUBDIRECTORIES.\n'
        should_remove = confirmation_cancel(msg)
        if should_remove:
            shutil.rmtree(img_dir_out)
            os.mkdir(img_dir_out)
    else:
        os.mkdir(img_dir_out)

    if args.ann_type == 'COCO':
        ann_path_out = os.path.join(args.out_dir, 'annotations.json')

        sliding_window_crop_coco(
            img_dir=args.image_dir,
            ann_path=args.ann_path,
            img_dir_out=img_dir_out,
            ann_path_out=ann_path_out,
            n_x=args.n_x,
            n_y=args.n_y,
            overlap=args.overlap,
            img_id=args.img_id,
            obj_id=args.obj_id,
            save_empty=not args.remove_empty,
        )

        msg = f'Sliding-window cropped images written to {img_dir_out}. '+\
            f'Sliding-window cropped annotation written to {ann_path_out}.'
        print(msg)

    elif args.ann_type == 'PVOC':
        ann_dir_out = os.path.join(args.out_dir, 'annotations')
        if os.path.exists(ann_dir_out):
            msg = f'Directory "{ann_dir_out} already exists. Should it be overwritten?"\n' +\
                f'WARNING: This will remove "{ann_dir_out}" and ALL SUBDIRECTORIES.\n'
            should_remove = confirmation_cancel(msg)
            if should_remove:
                shutil.rmtree(ann_dir_out)
                os.mkdir(ann_dir_out)
        else:
            os.mkdir(ann_dir_out)

        sliding_window_crop_pvoc(
            img_dir=args.image_dir,
            ann_dir=args.ann_path,
            img_dir_out=img_dir_out,
            ann_dir_out=ann_dir_out,
            n_x=args.n_x,
            n_y=args.n_y,
            overlap=args.overlap,
            save_empty=not args.remove_empty,
        )

        msg = f'Sliding-window cropped images written to {img_dir_out}. '+\
            f'Sliding-window cropped annotations written to {ann_dir_out}.'
        print(msg)
