''' Module for the ginjinn predict subcommand.
'''

import os
import sys
from ginjinn.ginjinn_config import GinjinnConfiguration
import ginjinn.ginjinn_config.config_error as config_error

def ginjinn_predict(args):
    '''ginjinn_predict

    GinJinn predict command.

    Parameters
    ----------
    args
        Parsed GinJinn commandline arguments for the ginjinn predict
        subcommand.
    '''

    project_dir = args.project_dir
    config_file = os.path.join(project_dir, 'ginjinn_config.yaml')

    if args.debug:
        config = GinjinnConfiguration.from_config_file(config_file)
    else:
        try:
            config = GinjinnConfiguration.from_config_file(config_file)
        except config_error.InvalidInputConfigurationError as err:
            print('\nInvalid input configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidModelConfigurationError as err:
            print('\nInvalid model configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidAugmentationConfigurationError as err:
            print('\nInvalid augmentation configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidGinjinnConfigurationError as err:
            print('\nInvalid GinJinn configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidOptionsConfigurationError as err:
            print('\nInvalid options configuration:')
            print(err)
            sys.exit(1)
        except config_error.InvalidTrainingConfigurationError as err:
            print('\nInvalid training configuration:')
            print(err)
            sys.exit(1)
        except Exception as any_e:
            raise any_e

    image_path = args.image_path

    # input
    img_dir = None
    img_names = None
    if os.path.isdir(image_path):
        img_dir = image_path
    else:
        img_names = [image_path]

    # output
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(config.project_dir, 'prediction')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    else:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    # class_names
    from ginjinn.data_reader.data_reader import get_class_names
    class_names = get_class_names(config.project_dir)

    # task
    task = config.task

    # predictor
    from detectron2.engine.defaults import DefaultPredictor
    predictor = DefaultPredictor(config.to_detectron2_config())

    # other
    save_cropped = args.save_cropped
    threshold = args.threshold
    padding = args.padding

    from ginjinn.predictor import predict_and_save
    predict_and_save(
        img_dir=img_dir,
        outdir=out_dir,
        class_names=class_names,
        task=task,
        predictor=predictor,
        save_cropped=save_cropped,
        threshold=threshold,
        crop_margin=padding,
        img_names=img_names,
    )

    # TODO implement prediction
