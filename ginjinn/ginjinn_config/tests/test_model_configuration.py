'''GinjinnModelConfiguration test module
'''

import pytest
import tempfile
import os
from ginjinn.ginjinn_config.model_config import GinjinnModelConfiguration, MODEL_NAMES
from ginjinn.ginjinn_config.config_error import InvalidModelConfigurationError

@pytest.fixture(scope='module', autouse=True)
def tmp_input_path():
    tmpdir = tempfile.TemporaryDirectory()

    tmp_file_path = os.path.join(tmpdir.name, 'weights.pkl')

    with open(tmp_file_path, 'w') as tmp_f:
        tmp_f.write('')

    yield tmp_file_path

    tmpdir.cleanup()

def test_simple_model(tmp_input_path):
    name = list(MODEL_NAMES.keys())[0]
    initial_weights = 'random'

    model = GinjinnModelConfiguration(
        name=name,
        initial_weights=initial_weights,
    )

    assert model.name == name
    assert model.initial_weights == initial_weights

    initial_weights = 'pretrained'
    model = GinjinnModelConfiguration(
        name=name,
        initial_weights=initial_weights,
    )

    assert model.name == name
    assert model.initial_weights == initial_weights

def test_invalid_model():
    name = 'some_invalid_name'
    initial_weights = 'random'

    with pytest.raises(InvalidModelConfigurationError):
        model = GinjinnModelConfiguration(
            name=name,
            initial_weights = 'random'
        )
    
    valid_name = list(MODEL_NAMES.keys())[0]
    with pytest.raises(InvalidModelConfigurationError):
        GinjinnModelConfiguration(name=valid_name, initial_weights='xyz')
