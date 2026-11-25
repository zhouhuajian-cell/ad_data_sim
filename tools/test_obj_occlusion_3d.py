import unittest
import numpy as np

from obj_occlusion_3d import foreground_depth, project_lidar, composite


class OcclusionTests(unittest.TestCase):
    def test_unknown_foreground_is_hidden(self):
        bg = np.zeros((3, 3, 3), np.uint8)
        fg = np.full((3, 3, 4), 255, np.uint8)
        z = np.full((3, 3), 10., np.float32)
        real = np.full((3, 3), np.nan, np.float32)
        mask = np.zeros((3, 3), bool)
        mask[1, 1] = True
        out, stats = composite(bg, fg, z, real, mask)
        self.assertEqual(int(out[1, 1, 0]), 0)
        self.assertEqual(int(out[0, 0, 0]), 255)
        self.assertEqual(stats['unknown_foreground_hidden'], 1)

    def test_projection_keeps_nearest(self):
        K = np.array([[10., 0., 5.], [0., 10., 5.], [0., 0., 1.]])
        xyz = np.array([[0., 0., 10.], [0., 0., 4.]])
        d = project_lidar(xyz, K, np.eye(3), np.zeros(3), (11, 11))
        self.assertEqual(float(d[5, 5]), 4.)

    def test_fill_stays_local_and_inside_mask(self):
        K = np.array([[10., 0., 5.], [0., 10., 5.], [0., 0., 1.]])
        mask = np.zeros((11, 11), bool)
        mask[4:7, 4:7] = True
        d, stat = foreground_depth(np.array([[0., 0., 5.]]), K,
                                   np.eye(3), np.zeros(3), mask, 2)
        self.assertTrue(np.isfinite(d[4:7, 4:7]).all())
        self.assertFalse(np.isfinite(d[0, 0]))
        self.assertEqual(stat["samples"], 1)


if __name__ == "__main__":
    unittest.main()
