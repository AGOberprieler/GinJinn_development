''' Module for setting up and handling argument parsers
'''

import argparse

def _setup_new_parser(subparsers):
    '''_setup_new_parser

    Setup parser for the ginjinn new subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the new subcommand.
    '''

    # TODO: implement

    parser = subparsers.add_parser(
        'new',
        help = '''
            Create a new GinJinn project.
        ''',
        description = '''
            Create a new GinJinn project.
        '''
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            Path to new GinJinn project directory.
        '''
    )

    return parser

def _setup_train_parser(subparsers):
    '''_setup_train_parser

    Setup parser for the ginjinn train subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the train subcommand.
    '''

    # TODO: implement

    parser = subparsers.add_parser(
        'train',
        help = '''
            Train a GinJinn model.
        ''',
        description = '''
            Train a GinJinn model.
        '''
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            Path to GinJinn project directory.
        '''
    )

    return parser

def _setup_evaluate_parser(subparsers):
    '''_setup_evaluate_parser

    Setup parser for the ginjinn evaluate subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the evaluate subcommand.
    '''

    # TODO: implement

    parser = subparsers.add_parser(
        'evaluate',
        aliases=['eval'],
        help = '''
            Evaluate a trained GinJinn model.
        ''',
        description = '''
            Evaluate a trained GinJinn model.
        '''
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            Path to GinJinn project directory.
        '''
    )

    return parser

def _setup_predict_parser(subparsers):
    '''_setup_predict_parser

    Setup parser for the ginjinn predict subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the predict subcommand.
    '''

    # TODO: implement

    parser = subparsers.add_parser(
        'predict',
        help = '''
            Predict from a trained GinJinn model.
        ''',
        description = '''
            Predict from a trained GinJinn model.
        '''
    )
    parser.add_argument(
        'project_dir',
        type = str,
        help = '''
            Path to GinJinn project directory.
        '''
    )

    return parser

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

        self._subparsers = self.parser.add_subparsers(
            dest='subcommand'
        )
        self._init_subparsers()

    def parse_args(self, args=None, namespace=None):
        '''parse_args
        Parses the commandline arguments and returns them in argparse
        format.

        Returns
        -------
        args
            Parsed argparse arguments
        '''
        print(self)
        return self.parser.parse_args(args=args, namespace=namespace)

    def _init_subparsers(self):
        '''_init_subparsers

        Initilialize parsers for GinJinn subcommands.
        '''

        _setup_new_parser(self._subparsers)
        _setup_train_parser(self._subparsers)
        _setup_evaluate_parser(self._subparsers)
        _setup_predict_parser(self._subparsers)

        # TODO: implement
