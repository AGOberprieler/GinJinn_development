'''GinjinnModelConfiguration test module
'''

import pytest
from ginjinn.ginjinn_config.model_config import GinjinnModelConfiguration, MODEL_NAMES
from ginjinn.ginjinn_config.config_error import InvalidGinjinnConfigurationError

def test_simple_model():
    name = MODEL_NAMES[0]
    learning_rate = 0.005
    batch_size = 1
    max_iter = 10000

    model = GinjinnModelConfiguration(
        name=name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_iter=max_iter,
    )

    assert model.name == name
    assert model.learning_rate == learning_rate
    assert model.batch_size == batch_size
    assert model.max_iter == max_iter

def test_invalid_model():
    name = 'some_invalid_name'
    learning_rate = 0.005
    batch_size = 1
    max_iter = 10000

    with pytest.raises(InvalidGinjinnConfigurationError):
        model = GinjinnModelConfiguration(
            name=name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_iter=max_iter,
        )