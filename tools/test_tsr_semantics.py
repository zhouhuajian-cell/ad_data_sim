import tempfile
import unittest
from pathlib import Path

from tsr_semantics import compatible, compatible_explicit, candidate_shape, semantic_name, valid_box
from tsr_param_preview import pick_material
from batch_walk import color_consistent
import numpy as np
import cv2


class SemanticTests(unittest.TestCase):
    def test_directions_and_families(self):
        self.assertTrue(compatible('mandatory_keep_left', 'mandatory_keep_left_v2.png'))
        self.assertFalse(compatible('mandatory_keep_left', 'mandatory_keep_right'))
        self.assertFalse(compatible('mandatory_turn_left', 'mandatory_keep_left'))
        self.assertFalse(compatible('mandatory', 'mandatory_keep_left'))
        self.assertTrue(compatible('speed_limit_70', 'speed_limit_80'))
        self.assertTrue(compatible('speed_limit', 'speed_limit_60'))
        self.assertFalse(compatible('prohibition_no_left', 'prohibition_no_u_turn'))
        self.assertTrue(compatible_explicit('mandatory', 'mandatory_keep_left'))
        self.assertFalse(compatible_explicit('mandatory_keep_left', 'mandatory_keep_right'))
        self.assertEqual(candidate_shape('unknown','circle',.4),('circle','class_inferred_review'))
        self.assertEqual(candidate_shape('unknown','circle',.3)[0],'unknown')
        self.assertEqual(semantic_name('TH_mandatory_keep_left_v3.png', 'TH'), 'mandatory_keep_left')

    def test_boxes(self):
        self.assertTrue(valid_box([40, 40, 80, 80], (1080, 1920, 3), 14))
        for box in ([-1,0,30,30], [0,0,5000,500], [10,10,12,12], [0,0,float('nan'),20]):
            self.assertFalse(valid_box(box, (1080, 1920, 3), 14))

    def test_speed_sign_rejects_blue_or_yellow_conflicts(self):
        blue=np.full((24,24,3),(255,0,0),np.uint8)
        yellow=np.full((24,24,3),(0,255,255),np.uint8)
        self.assertFalse(color_consistent(blue,blue,'speed_limit',.04,.06)[0])
        self.assertFalse(color_consistent(yellow,yellow,'speed_limit',.04,.06)[0])

    def test_preview_exact_country_and_semantics(self):
        with tempfile.TemporaryDirectory() as d:
            th = Path(d) / 'TH'
            th.mkdir()
            for name in ('mandatory_keep_left_v2.png', 'mandatory_keep_right.png'):
                (th / name).touch()
            self.assertEqual(Path(pick_material(d, 'TH', 'mandatory_keep_left')).name, 'mandatory_keep_left_v2.png')
            self.assertIsNone(pick_material(d, 'TH', 'mandatory_keep_left', 'mandatory_keep_right.png'))
            self.assertIsNone(pick_material(d, 'TH', 'mandatory'))
            self.assertIsNone(pick_material(d, 'BR', 'mandatory_keep_left'))
            self.assertIsNone(pick_material(d, 'TH', ''))


if __name__ == '__main__':
    unittest.main()
