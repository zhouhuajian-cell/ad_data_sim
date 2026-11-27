import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from tsr_sign_render import render_sign


class CoverageTest(unittest.TestCase):
    def check(self, result_bbox):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        fake = types.ModuleType('pipeline6')
        fake.run_pipeline6 = lambda *a, **k: (image + 1, {'ok': True, 'bbox': result_bbox})
        with patch.dict(sys.modules, {'pipeline6': fake}):
            return image, render_sign(image, [10, 10, 32, 33], None, 'circle')

    def test_rejects_shrunken_render(self):
        image, (out, info) = self.check([10, 12, 29, 27])
        self.assertFalse(info['ok'])
        self.assertEqual(info['reason'], 'refined_bbox_undercoverage')
        np.testing.assert_array_equal(image, out)

    def test_accepts_comparable_render(self):
        _, (out, info) = self.check([10, 10, 32, 33])
        self.assertTrue(info['ok'])
        self.assertEqual(int(out[0, 0, 0]), 1)

if __name__ == '__main__':
    unittest.main()
