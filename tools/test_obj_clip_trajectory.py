import unittest
import numpy as np

from obj_clip_trajectory import create_track, pose_at, stable_seed


class TrajectoryTests(unittest.TestCase):
    def test_stable_seed(self):
        self.assertEqual(stable_seed("a", 1), stable_seed("a", 1))
        self.assertNotEqual(stable_seed("a", 1), stable_seed("b", 1))

    def test_static_world_object_moves_back_in_ego(self):
        stems = ["0.0", "1.0"]
        poses = {stems[0]: np.eye(4), stems[1]: np.eye(4)}
        poses[stems[1]][0, 3] = 10.0
        tr = create_track("x", "cone", (20, 1), 0, poses, stems)
        self.assertAlmostEqual(pose_at(tr, poses[stems[1]], stems[1])[0], 10.)

    def test_equal_speed_vehicle_stays_relative(self):
        stems = ["0.0", "1.0"]
        poses = {stems[0]: np.eye(4), stems[1]: np.eye(4)}
        poses[stems[1]][0, 3] = 10.0
        tr = create_track("x", "car", (20, 1), 180, poses, stems,
                          relative_speed_mps=0.0)
        x, y, yaw = pose_at(tr, poses[stems[1]], stems[1])
        self.assertAlmostEqual(x, 20.)
        self.assertAlmostEqual(y, 1.)
        self.assertAlmostEqual(abs(yaw), 180.)


if __name__ == "__main__":
    unittest.main()
