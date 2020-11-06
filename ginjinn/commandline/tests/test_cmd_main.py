import pytest
import sys
import copy
import tempfile
import os
import mock
import pkg_resources
import yaml

from ginjinn.commandline import main, commandline_app, argument_parser
from ginjinn.commandline import splitter
from ginjinn.commandline import simulate
from ginjinn.commandline import train

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

@pytest.fixture(scope='module', autouse=True)
def example_config(tmp_dir, simulate_coco):
    img_dir, ann_path = simulate_coco

    example_config_1_path = pkg_resources.resource_filename(
        'ginjinn', 'data/ginjinn_config/example_config_1.yaml',
    )

    with open(example_config_1_path) as config_f:
        config = yaml.load(config_f)

    config['input']['train']['annotation_path'] = ann_path
    config['input']['train']['image_path'] = img_dir

    config_dir = os.path.join(tmp_dir, 'example_config')
    os.mkdir(config_dir)

    config_file = os.path.join(config_dir, 'config_0.yaml')
    with open(config_file, 'w') as config_f:
        yaml.dump(config, config_f)

    return (config, config_file)

@pytest.fixture(scope='module', autouse=True)
def example_project(tmp_dir, example_config):
    config, _ = example_config

    project_dir = os.path.join(tmp_dir, 'example_project')
    os.mkdir(project_dir)

    config_file = os.path.join(project_dir, 'ginjinn_config.yaml')
    with open(config_file, 'w') as config_f:
        yaml.dump(config, config_f)
    
    return project_dir

def test_main_simple(tmp_dir):
    project_dir = os.path.join(tmp_dir, 'test_new_0')

    tmp = copy.deepcopy(sys.argv)
    sys.argv = ['ginjinn', 'new', project_dir]
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


def test_simulate(tmp_dir):
    simulate_dir = os.path.join(tmp_dir, 'test_simulate_0')

    args = argument_parser.GinjinnArgumentParser().parse_args(
        [
            'simulate',
            'shapes',
            '-o', simulate_dir,
            '-n', '5',
        ]
    )

    simulate.ginjinn_simulate(args)

    with mock.patch('builtins.input', lambda *args: 'y'):
        simulate.ginjinn_simulate(args)

def test_train(example_project):
    project_dir = example_project
    args = argument_parser.GinjinnArgumentParser().parse_args(
        [
            'train',
            project_dir
        ]
    )

    train.ginjinn_train(args)
