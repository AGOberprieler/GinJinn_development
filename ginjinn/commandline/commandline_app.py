''' GinJinn commandline application code.
'''

from .argument_parser import GinjinnArgumentParser

class GinjinnCommandlineApplication():
    '''GinjinnCommandlineApplication

    GinJinn commandline application.
    '''
    def __init__(self):
        self.parser = GinjinnArgumentParser()
        self.args = None

    def run(self):
        '''run
        Start GinJinn commandline application.
        '''
        self.args = self.parser.parse_args()
