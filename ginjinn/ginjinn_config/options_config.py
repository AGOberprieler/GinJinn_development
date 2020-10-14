'''
GinJinn options configuration module
'''

import copy
import os
# from typing import Optional
from .config_error import InvalidOptionsConfigurationError

N_CORES = os.cpu_count()

class GinjinnOptionsConfiguration: #pylint: disable=too-few-public-methods
    '''A class representing GinJinn model configurations.

    Parameters
    ----------
    resume : bool
        Determines, whether a previous run should be resumed
    n_threads : int
        Number of CPU threads to use.
    '''
    def __init__( #pylint: disable=too-many-arguments
        self,
        resume: bool,
        n_threads: int,
    ):
        self.resume = resume
        self.n_threads = n_threads

        self._check_config()

    @classmethod
    def from_dictionary(cls, config: dict):
        '''Build GinjinnOptionsConfiguration from a dictionary object.

        Parameters
        ----------
        config : dict
            Dictionary object containing the options configuration.

        Returns
        -------
        GinjinnOptionsConfiguration
            GinjinnOptionsConfiguration constructed with the configuration
            given in config.
        '''

        default_config = {
            'resume': False,
            'n_threads': N_CORES - 1 if N_CORES > 1 else N_CORES,
        }

        # Maybe implement this more elegantly...
        default_config.update(config)
        config = copy.deepcopy(default_config)

        return cls(
            resume=config['resume'],
            n_threads=config['n_threads'],
        )

    def _check_n_threads(self):
        ''' Check n_threads config

        Raises
        ------
        InvalidOptionsConfigurationError
            Raised if n_threads value is invalid.
        '''
        if self.n_threads < 1:
            raise InvalidOptionsConfigurationError(
                'n_threads must be a positive number.'
            )

    def _check_config(self):
        ''' Check configuration values for validity.
        '''
        self._check_n_threads()
