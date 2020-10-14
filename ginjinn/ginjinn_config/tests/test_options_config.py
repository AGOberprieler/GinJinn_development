''' Module for testing options_config.py
'''

import pytest
from ginjinn.ginjinn_config.options_config import GinjinnOptionsConfiguration
from ginjinn.ginjinn_config.config_error import InvalidOptionsConfigurationError

def test_simple():
    n_threads=1
    resume=True
    options_0 = GinjinnOptionsConfiguration(
        resume=resume,
        n_threads=n_threads
    )
    assert options_0.resume == resume
    assert options_0.n_threads == n_threads

    options_1 = GinjinnOptionsConfiguration.from_dictionary({
        'n_threads': n_threads,
        'resume': resume,
    })
    assert options_1.resume == resume
    assert options_1.n_threads == n_threads

def test_invalid():
    n_threads=0
    resume=True

    with pytest.raises(InvalidOptionsConfigurationError):
        GinjinnOptionsConfiguration(
            resume=resume,
            n_threads=n_threads
        )
