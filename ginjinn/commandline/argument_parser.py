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
    # parser.add_argument(
    #     '-nr', '--no_resume',
    #     type = bool,
    #     help = '''
    #         Do not resume training. If this option is set, training will
    #         start from scratch, discarding previous training checkpoints
    #         PERMANENTLY.
    #     ''',
    #     # action='store_true',
    #     default=None,
    # )

    parser.add_argument('-r', '--resume', dest='resume', action='store_true')
    parser.add_argument('-nr', '--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=None)

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

def _setup_split_parser(subparsers):
    '''_setup_split_parser

    Setup parser for the ginjinn split subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the split subcommand.
    '''

    # TODO: implement

    parser = subparsers.add_parser(
        'split',
        help = '''
            Split dataset (images and annotations) into test, train, and optionally
            evaluation datasets.
        ''',
        description = '''
            Split dataset (images and annotations) into test, train, and optionally
            evaluation datasets.
        '''
    )
    required_parser = parser.add_argument_group('required named arguments')
    required_parser.add_argument(
        '-i', '--image_dir',
        type = str,
        help = '''
            Path to directory containing images.
        ''',
        required = True,
    )
    required_parser.add_argument(
        '-a', '--annotation_path',
        type = str,
        help = '''
            Path to directory containing annotations (PVOC) or path to an annotation
            JSON file (COCO).
        ''',
        required = True,
    )
    required_parser.add_argument(
        '-o', '--output_dir',
        type = str,
        help = '''
            Path to output directory. Splits will be written to output_dir/train,
            output_dir/test, and output_dir/eval, respectively. The output directory
            will be created, if it does not exist. 
        ''',
        required = True,
    )
    required_parser.add_argument(
        '-d', '--task',
        type = str,
        choices = [
            'instance-segmentation', 'bbox-detection'
        ],
        help = '''
            Task, which the dataset will be used for.
        ''',
        required = True,
    )
    required_parser.add_argument(
        '-k', '--ann_type',
        type = str,
        choices = ['COCO', 'PVOC'],
        help = '''
            Dataset type.
        ''',
        required = True,
    )
    # parser.add_argument(
    #     '-t', '--train_fraction',
    #     type = float,
    #     help = '''
    #         Fraction of the dataset to use for training. (Default: 0.6)
    #     ''',
    #     default = 0.6,
    # )
    optional_parser = parser.add_argument_group('optional arguments')
    optional_parser.add_argument(
        '-t', '--test_fraction',
        type = float,
        help = '''
            Fraction of the dataset to use for testing. (Default: 0.2)
        ''',
        default = 0.2,
    )
    optional_parser.add_argument(
        '-v', '--validation_fraction',
        type = float,
        help = '''
            Fraction of the dataset to use for validation while training. (Default: 0.2)
        ''',
        default = 0.2,
    )

    return parser

def _setup_simulate_parser(subparsers):
    '''_setup_simulate_parser

    Setup parser for the ginjinn simulate subcommand.

    Parameters
    ----------
    subparsers
        An object returned by argparse.ArgumentParser.add_subparsers()

    Returns
    -------
    parser
        An argparse ArgumentParser, registered for the simulate subcommand.
    '''

    # TODO: implement

    parser = subparsers.add_parser(
        'simulate',
        help = '''
            Simulate datasets.
        ''',
        description = '''
            Simulate datasets.
        ''',
    )
    simulate_parsers = parser.add_subparsers(
        dest='simulate_subcommand',
        help='Types of simulations.'
    )

    # == shapes
    shapes_parser = simulate_parsers.add_parser(
        'shapes',
        help = '''
            Simulate a simple segmentation dataset with COCO annotations,
            or a simple bounding-box dataset with PVOC annotations,
            containing two classes: circles and triangles.
        ''',
        description = '''
            Simulate a simple segmentation dataset with COCO annotations,
            or a simple bounding-box dataset with PVOC annotations,
            containing two classes: circles and triangles.
        ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    required = shapes_parser.add_argument_group('required arguments')
    required.add_argument(
        '-o', '--out_dir',
        type = str,
        help = '''
            Path to directory, which the simulated data should be written to.
        ''',
        required=True,
    )

    optional = shapes_parser.add_argument_group('optional arguments')
    optional.add_argument(
        '-a', '--ann_type',
        type = str,
        help = '''
            Type of annotations to simulate.
        ''',
        choices=['COCO', 'PVOC'],
        default='COCO',
    )
    optional.add_argument(
        '-n', '--n_images',
        type = int,
        help = '''
            Number of images to simulate.
        ''',
        default=100,
    )
    optional.add_argument(
        '-w0', '--min_w',
        type = int,
        help = '''
            Minimum image width.
        ''',
        default=400,
    )
    optional.add_argument(
        '-w1', '--max_w',
        type = int,
        help = '''
            Maximum image width.
        ''',
        default=400,
    )
    optional.add_argument(
        '-h0', '--min_h',
        type = int,
        help = '''
            Minimum image height.
        ''',
        default=400,
    )
    optional.add_argument(
        '-h1', '--max_h',
        type = int,
        help = '''
            Maximum image height.
        ''',
        default=400,
    )
    optional.add_argument(
        '-n0', '--min_n_shapes',
        type = int,
        help = '''
            Minimum number of shapes per image.
        ''',
        default=1,
    )
    optional.add_argument(
        '-n1', '--max_n_shapes',
        type = int,
        help = '''
            Maximum number of shapes per image.
        ''',
        default=4,
    )
    optional.add_argument(
        '-t', '--triangle_prob',
        type = float,
        help = '''
            Probability of generating a triangle. Default is 0.5, meaning that
            triangles and circle are equally represented.
        ''',
        default=0.5,
    )
    optional.add_argument(
        '-ccol', '--circle_col',
        type = str,
        help = '''
            Mean circle color as Hex color code.
        ''',
        default='#C87D7D',
    )
    optional.add_argument(
        '-tcol', '--triangle_col',
        type = str,
        help = '''
            Mean triangle color as Hex color code.
        ''',
        default='#7DC87D',
    )
    optional.add_argument(
        '-cvar', '--color_variance',
        type = float,
        help = '''
            Variance around mean shape colors.
        ''',
        default=0.15,
    )
    optional.add_argument(
        '-mnr', '--min_shape_radius',
        type = float,
        help = '''
            Minimum shape radius.
        ''',
        default=25.0,
    )
    optional.add_argument(
        '-mxr', '--max_shape_radius',
        type = float,
        help = '''
            Maximum shape radius.
        ''',
        default=75.0,
    )
    optional.add_argument(
        '-mna', '--min_shape_angle',
        type = float,
        help = '''
            Minimum shape rotation in degrees.
        ''',
        default=0.0,
    )
    optional.add_argument(
        '-mxa', '--max_shape_angle',
        type = float,
        help = '''
            Maximum shape rotation in degrees.
        ''',
        default=60.0,
    )
    optional.add_argument(
        '-b', '--noise',
        type = float,
        help = '''
            Amount of noise to add.
        ''',
        default=0.005,
    )

    # ==
    # ... further simulations ...
    # ==

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
            description=self._description,
        )
        self.parser.add_argument(
            '-d', '--debug',
            help='Debug mode',
            action='store_true',
        )

        self._subparsers = self.parser.add_subparsers(
            dest='subcommand',
            help='GinJinn subcommands.'
        )
        self._init_subparsers()

    def parse_args(self, args=None, namespace=None):
        '''parse_args
        Parses the commandline arguments and returns them in argparse
        format.

        Parameters
        ----------
        args
            List of strings to parse. If None, the strings are taken from sys.argv.
        namespace
            An object to take the attributes. The default is a new empty argparse Namespace object.

        Returns
        -------
        args
            Parsed argparse arguments
        '''

        return self.parser.parse_args(args=args, namespace=namespace)

    def _init_subparsers(self):
        '''_init_subparsers

        Initilialize parsers for GinJinn subcommands.
        '''

        _setup_new_parser(self._subparsers)
        _setup_train_parser(self._subparsers)
        _setup_evaluate_parser(self._subparsers)
        _setup_predict_parser(self._subparsers)
        _setup_split_parser(self._subparsers)
        _setup_simulate_parser(self._subparsers)

        # TODO: implement
