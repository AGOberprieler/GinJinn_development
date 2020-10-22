''' Module for setting up and handling argument parsers
'''

import argparse

# TODO: implement
class GinjinnArgumentParser():
    '''GinjinnArgumentParser

    Class for setting up and handling commandline arguments.
    '''
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        '''parse_args
        Parses the commandline arguments and returns them in argparse 
        format.

        Returns
        -------
        args
            Parsed argparse arguments
        '''
        return self.parser.parse_args()
