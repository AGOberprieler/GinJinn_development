''' GinJinn split commandline module
'''

import sys
import pandas as pd
import numpy as np


from ginjinn.data_reader.data_splitter import create_split_2

def confirmation_cancel(question: str) -> bool:
    '''Ask question expecting 'yes' or 'no'.

    Parameters
    ----------
    question : str
        Question to be printed

    Returns
    -------
    bool
        True or False for 'yes' or 'no', respectively
    '''
    valid = {'yes': True, 'y': True, 'no': False, 'n': False}
    cancel = ['c', 'cancel', 'quit', 'q']

    while True:
        choice = input(question + ' [y(es)/n(o)/c(ancel)]\n').strip().lower()
        if choice in valid.keys():
            return valid[choice]
        elif choice in cancel:
            sys.exit()
        print('Please type "yes" or "no" (or "cancel")\n')

def on_split_dir_exists(split_dir: str) -> bool:
    '''on_split_dir_exists

    Callback for when the split output directory already exists.

    Parameters
    ----------
    split_dir : str
        Path to split directory.

    Returns
    -------
    bool
        Whether the existing directory should be overwritten.
    '''
    return confirmation_cancel(
        '"' + split_dir + '" already exists.\nDo you want do overwrite it? '+\
        'ATTENTION: This will DELETE "' + split_dir + '" and all subdirectories.'
    )

def on_split_proposal(split_df: 'pd.DataFrame') -> bool:
    '''on_split_proposal

    Callback for proposing a split.

    Parameters
    ----------
    split_df : 'pd.DataFrame'
        pandas.DataFrame containing split information.

    Returns
    -------
    bool
        Whether the proposed split should be accepted.
    '''

    df_pretty = pd.DataFrame(
        [[f'{a} ({round(b, 2)})' for a,b in zip(r, r / r.sum())] for _, r in split_df.iterrows()],
        columns=split_df.columns,
        index=split_df.index
    )

    print('\nSplit proposal:')
    print(df_pretty)
    return confirmation_cancel(
        '\nDo you want to accept this split? (Otherwise a new one will be generated.)'
    )

def on_no_valid_split() -> bool:
    '''on_no_valid_split

    Callback for when no valid split was found.


    Returns
    -------
    bool
        Whether another try for finding a valid split should be made.
    '''

    return confirmation_cancel(
        'Could not find a valid split. Try again?'
    )


def ginjinn_split(args):
    '''ginjinn_split

    GinJinn split command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn split
        subcommand.
    '''

    # print('Running ginjinn split')
    # print(args)

    ann_path = args.annotation_path
    img_dir = args.image_dir
    split_dir = args.output_dir
    task = args.task
    ann_type = args.ann_type

    p_val = args.validation_fraction
    p_test = args.test_fraction

    create_split_2(
        ann_path=ann_path,
        img_path=img_dir,
        split_dir=split_dir,
        task=task,
        ann_type=ann_type,
        p_val=p_val,
        p_test=p_test,
        on_split_dir_exists=on_split_dir_exists,
        on_split_proposal=on_split_proposal,
        on_no_valid_split=on_no_valid_split,
    )

    print(f'Datasets written to "{split_dir}".')
