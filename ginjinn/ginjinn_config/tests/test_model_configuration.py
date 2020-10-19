'''GinjinnModelConfiguration test module
'''

import pytest
from ginjinn.ginjinn_config.model_config import GinjinnModelConfiguration, MODEL_NAMES
from ginjinn.ginjinn_config.config_error import InvalidModelConfigurationError

def test_simple_model():
    name = list(MODEL_NAMES.keys())[0]

    model = GinjinnModelConfiguration(
        name=name,
    )

    assert model.name == name

def test_invalid_model():
    name = 'some_invalid_name'


    with pytest.raises(InvalidModelConfigurationError):
        model = GinjinnModelConfiguration(
            name=name,
        )
