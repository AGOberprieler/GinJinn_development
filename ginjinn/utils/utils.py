"""
Module for generic helper functions.
"""

import sys

def confirmation(question: str) -> bool:
    """Ask question expecting "yes" or "no".

    Parameters
    ----------
    question : str
        Question to be printed

    Returns
    -------
    bool
        True or False for "yes" or "no", respectively
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}

    while True:
        choice = input(question + " [y/n]\n").strip().lower()
        if choice in valid.keys():
            return valid[choice]
        print("Please type 'yes' or 'no'\n")

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
