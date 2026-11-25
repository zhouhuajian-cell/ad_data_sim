import tempfile
import unittest
from pathlib import Path

from tsr_semantics import compatible, semantic_name, valid_box
from tsr_param_preview import pick_material


class SemanticTests(unittest.TestCase):
    def test_directions_and_families(self):
        self.assertTrue(compatible('mandatory_keep_left', 'mandatory_keep_left_v2.png'))
        self.assertFalse(compatible('mandatory_keep_left', 'mandatory_keep_right'))
        self.assertFalse(compatible('mandatory_turn_left', 'mandatory_keep_left'))
        self.assertFalse(compatible('mandatory', 'mandatory_keep_left'))
        self.assertTrue(compatible('speed_limit_70', 'speed_limit_80'))
        self.assertTrue(compatible('speed_limit', 'speed_limit_60'))
        self.assertFalse(compatible('prohibition_no_left', 'prohibition_no_u_turn'))
        self.assertEqual(semantic_name('TH_mandatory_keep_left_v3.png', 'TH'), 'mandatory_keep_left')

    def test_boxes(self):
        self.assertTrue(valid_box([40, 40, 80, 80], (1080, 1920, 3), 14))
        for box in ([-1,0,30,30], [0,0,5000,500], [10,10,12,12], [0,0,float('nan'),20]):
            self.assertFalse(valid_box(box, (1080, 1920, 3), 14))

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
