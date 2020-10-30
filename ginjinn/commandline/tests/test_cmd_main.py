import pytest
import sys
import copy
import tempfile
import os
import mock

from ginjinn.commandline import main, commandline_app, argument_parser, splitter
from ginjinn.simulation import generate_simple_shapes_coco

@pytest.fixture(scope='module', autouse=True)
def tmp_dir():
    tmpdir = tempfile.TemporaryDirectory()

    yield tmpdir.name

    tmpdir.cleanup()

@pytest.fixture(scope='module')
def simulate_coco(tmp_dir):
    sim_dir = os.path.join(tmp_dir, 'sim_coco')
    os.mkdir(sim_dir)

    img_dir = os.path.join(sim_dir, 'images')
    os.mkdir(img_dir)
    ann_path = os.path.join(sim_dir, 'annotations.json')
    generate_simple_shapes_coco(
        img_dir=img_dir, ann_file=ann_path, n_images=40,
    )
    return img_dir, ann_path


def test_main_simple():
    tmp = copy.deepcopy(sys.argv)
    sys.argv = ['ginjinn', 'new', 'my_project_dir']
    main()
    sys.argv = tmp

def test_splitting(tmp_dir, simulate_coco):
    img_dir, ann_path = simulate_coco


    split_dir = os.path.join(tmp_dir, 'test_splitting_0')
    os.mkdir(split_dir)

    args = argument_parser.GinjinnArgumentParser().parse_args(
        [
            'split',
            '-i', img_dir,
            '-a', ann_path,
            '-o', split_dir,
            '-d', 'instance-segmentation',
            '-k', 'COCO'
        ]
    )

    def y_gen():
        while True:
            yield 'y'
    y_it = y_gen()

    def y(*args, **kwargs):
        return next(y_it)
    
    with mock.patch('builtins.input', y):
        splitter.ginjinn_split(args)

    with mock.patch('builtins.input', y):
        splitter.ginjinn_split(args)
    
    with mock.patch('builtins.input', lambda *args: 'n'):
        splitter.ginjinn_split(args)
