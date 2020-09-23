''' Tests for data simulation
'''

import os
import tempfile
import pytest

from ginjinn.simulation import generate_simple_shapes_coco

def test_simple_shapes_coco():
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = os.path.join(tmp_dir, 'images')
        os.mkdir(img_dir)

        ann_file = os.path.join(tmp_dir, 'annotations.json')
        generate_simple_shapes_coco(
            tmp_dir,
            ann_file,
            n_images=10
        )

        generate_simple_shapes_coco(
            tmp_dir,
            ann_file,
            n_images=10,
            min_rot=0, max_rot=0
        )