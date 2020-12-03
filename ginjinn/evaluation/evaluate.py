''' Evaluation module
'''

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model


def evaluate(
    cfg: CfgNode,
    task: str,
    dataset: str = "test"
    ):
    """Evaluate registered test dataset using COCOEvaluator

    Parameters
    ----------
    cfg : CfgNode
        Detectron2 configuration
    task : str
        "bbox-detection" or "instance-segmentation"
    dataset : str
        Name of registered dataset

    Returns
    -------
    eval_results : OrderedDict
        AP values
    """
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(
        model,
        save_dir = cfg.OUTPUT_DIR
    )
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

    if task == "bbox-detection":
        eval_tasks = ("bbox", )
    if task == "instance-segmentation":
        eval_tasks = ("bbox", "segm")

    evaluator = COCOEvaluator(
        dataset,
        tasks = eval_tasks,
        distributed = False,
        output_dir = cfg.OUTPUT_DIR
    )
    test_loader = build_detection_test_loader(cfg, dataset)
    eval_results = inference_on_dataset(model, test_loader, evaluator)
    return eval_results
