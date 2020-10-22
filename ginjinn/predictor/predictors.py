''' Predictor module
'''

from detectron2.engine import DefaultPredictor

class GinjinnPredictor(DefaultPredictor):
    '''GinjinnPredictor
        A class for predicting from a trained Detectron2 model.

        Parameters
        ----------
        cfg : Object
            Detectron2 configuration object
        replicates : int, optional
            Number of replicated predictions to conduct for each image,
            by default 1
    '''
    def __init__(
        self,
        cfg,
        replicates: int = 1
    ):
        super().__init__(cfg)
        self.replicates = replicates # TODO: thin about whether we need this
