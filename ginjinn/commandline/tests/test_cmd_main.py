import pytest
import sys
import copy

from ginjinn.commandline import main

def test_main_simple():
    tmp = copy.deepcopy(sys.argv)
    sys.argv = ['ginjinn', 'new', 'my_project_dir']
    main()
    sys.argv = tmp