''' Module for the ginjinn info subcommand
'''

def ginjinn_info(args):
    '''ginjinn_info

    GinJinn info command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn
        info subcommand.
    '''
    from ginjinn.utils.utils import dataset_info

    dataset_info(
        ann_path=args.ann_path,
        img_dir=args.img_dir,
        ann_type=args.ann_type,
    )
