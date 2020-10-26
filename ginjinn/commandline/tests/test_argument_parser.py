import pytest
import subprocess

from ginjinn.commandline import GinjinnArgumentParser

def test_simple_():
    # new
    p = GinjinnArgumentParser()
    args = p.parse_args(['new', 'my_project_dir'])

    # train
    args = p.parse_args(['train', 'my_project_dir'])

    # evaluate
    args = p.parse_args(['evaluate', 'my_project_dir'])

    # predict
    args = p.parse_args(['predict', 'my_project_dir'])