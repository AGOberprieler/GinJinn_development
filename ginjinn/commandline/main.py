''' Commandline main
'''

from .argument_parser import GinjinnArgumentParser

def main():
    '''main
    GinJinn main.
    '''
    parser = GinjinnArgumentParser()
    args = parser.parse_args()
    print(args)
    print('GinJinn called!')
