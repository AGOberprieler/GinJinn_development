''' GinJinn commandline application code.
'''

from .argument_parser import GinjinnArgumentParser
from .splitter import ginjinn_split
from .simulate import ginjinn_simulate
from .new import ginjinn_new
from .train import ginjinn_train
from .utils import ginjinn_utils
from .evaluate import ginjinn_evaluate
from .predict import ginjinn_predict
from .info import ginjinn_info

class GinjinnCommandlineApplication():
    '''GinjinnCommandlineApplication

    GinJinn commandline application.
    '''
    def __init__(self):
        self.parser = GinjinnArgumentParser()
        self.args = None

    def run(self, args=None, namespace=None):
        '''run
        Start GinJinn commandline application.

        Parameters
        ----------
        args
            List of strings to parse. If None, the strings are taken from sys.argv.
        namespace
            An object to take the attributes. The default is a new empty argparse Namespace object.
        '''
        self.args = self.parser.parse_args(args=args, namespace=namespace)
        # print(self.args)

        if self.args.subcommand == 'new':
            self._run_new()
        elif self.args.subcommand == 'split':
            self._run_split()
        elif self.args.subcommand == 'simulate':
            self._run_simulate()
        elif self.args.subcommand == 'train':
            self._run_train()
        elif self.args.subcommand == 'utils':
            self._run_utils()
        elif self.args.subcommand == 'evaluate':
            self._run_evaluate()
        elif self.args.subcommand == 'predict':
            self._run_predict()
        elif self.args.subcommand == 'info':
            self._run_info()

    def _run_split(self):
        '''_run_split
        Run the GinJinn split command.
        '''

        ginjinn_split(self.args)

    def _run_simulate(self):
        '''_run_simulate
        Run the GinJinn simulate command.
        '''

        ginjinn_simulate(self.args)

    def _run_new(self):
        '''_run_new
        Run the GinJinn new command.
        '''

        ginjinn_new(self.args)

    def _run_train(self):
        '''_run_train
        Run the GinJinn train command.
        '''

        ginjinn_train(self.args)

    def _run_utils(self):
        '''_run_utils
        Run the GinJinn utils command.
        '''

        ginjinn_utils(self.args)

    def _run_evaluate(self):
        '''_run_evaluate
        Run the GinJinn evaluate command.
        '''

        ginjinn_evaluate(self.args)

    def _run_predict(self):
        '''_run_predict
        Run the GinJinn predict command.
        '''

        ginjinn_predict(self.args)

    def _run_info(self):
        '''_run_info
        Run the GinJinn info command.
        '''

        ginjinn_info(self.args)
