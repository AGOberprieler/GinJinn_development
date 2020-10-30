''' GinJinn commandline application code.
'''

from .argument_parser import GinjinnArgumentParser
from .splitter import ginjinn_split

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

    def _run_split(self):
        '''_run_split
        Run the GinJinn split command.
        '''

        ginjinn_split(self.args)

    def _run_new(self):
        '''_run_new
        Run the GinJinn new command.
        '''

        print('Running ginjinn new')
        print(self.args)

        # TODO: implement
