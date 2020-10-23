''' Module for setting up and handling argument parsers
'''

import argparse

# TODO: implement

# Note: It is a deliberate decision not to subclass argparse.ArgumentParser.
#       It might be preferable to work with composition instead of inheritance,
#       since it might be desirable to include postprocessing steps after argparse
#       parsing.
class GinjinnArgumentParser():
    '''GinjinnArgumentParser

    Class for setting up and handling commandline arguments.
    '''

    _description = '''
        GinJinn is a framework for simplifying the setup, training, evaluation,
        and deployment of object detection models.
    '''

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description=self._description
        )

        self._subparsers = self.parser.add_subparsers()

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

    def _init_subparsers(self):
        '''_init_subparsers

        Initilialize parsers for GinJinn subcommands.
        '''
        # TODO: implement
