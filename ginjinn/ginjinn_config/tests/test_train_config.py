'''GinjinnTrainingConfiguration test module
'''

import pytest
from ginjinn.ginjinn_config.training_config import GinjinnTrainingConfiguration
from ginjinn.ginjinn_config.config_error import InvalidTrainingConfigurationError

@pytest.fixture
def simple_config():
    return {
        "learning_rate": 0.002,
        "batch_size": 1,
        "max_iter": 40000,
    }

def test_simple_training_config(simple_config):
    learning_rate = simple_config['learning_rate']
    batch_size = simple_config['batch_size']
    max_iter = simple_config['max_iter']

    training = GinjinnTrainingConfiguration(
        learning_rate = learning_rate,
        batch_size = batch_size,
        max_iter = max_iter,
    )

    assert training.learning_rate == learning_rate
    assert training.batch_size == batch_size
    assert training.max_iter == max_iter

def test_invalid_training_config(simple_config):
    learning_rate = simple_config['learning_rate']
    batch_size = simple_config['batch_size']
    max_iter = simple_config['max_iter']

    with pytest.raises(InvalidTrainingConfigurationError):
        training = GinjinnTrainingConfiguration(
            learning_rate = -1,
            batch_size = batch_size,
            max_iter = max_iter,
        )
    
    with pytest.raises(InvalidTrainingConfigurationError):
        training = GinjinnTrainingConfiguration(
            learning_rate = learning_rate,
            batch_size = 0,
            max_iter = max_iter,
        )
    
    with pytest.raises(InvalidTrainingConfigurationError):
        training = GinjinnTrainingConfiguration(
            learning_rate = learning_rate,
            batch_size = batch_size,
            max_iter = -1,
        )

def test_from_dictionary(simple_config):
    training = GinjinnTrainingConfiguration.from_dictionary(
        simple_config
    )

    assert training.learning_rate == simple_config['learning_rate']
    assert training.batch_size == simple_config['batch_size']
    assert training.max_iter == simple_config['max_iter']
