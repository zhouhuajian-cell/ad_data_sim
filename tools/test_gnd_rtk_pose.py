import tempfile
import unittest
from pathlib import Path
import numpy as np
from gnd_rtk_pose import load_body_poses


class RtkPoseTest(unittest.TestCase):
    def test_interpolates_and_rejects_disagreement(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'rtkimu_qj_7683';p.mkdir();ext=root/'parameter_yaml'/'v1'/'extrinsic';ext.mkdir(parents=True)
            (ext/'imu-1_to_body-2.yaml').write_text('transformation: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]',encoding='utf8')
            data=np.array([[100,5,4,0,0,0,0,1],[100.1,6,4,0,0,0,0,1],[100.2,7,4,0,0,0,0,1]])
            np.savetxt(p/'Vehicle.pose',data,delimiter=',')
            times=[100.05,100.15];icp=np.tile(np.eye(4),(2,1,1));icp[1,0,3]=1
            poses,info=load_body_poses(d,times,icp,max_time_gap=.11)
            self.assertAlmostEqual(poses[1,0,3],1,places=6)
            self.assertFalse(info['pixel_alignment_verified'])
            icp[1,0,3]=5
            with self.assertRaisesRegex(ValueError,'disagree'):
                load_body_poses(d,times,icp,max_time_gap=.11)
            with self.assertRaisesRegex(ValueError,'cover'):
                load_body_poses(d,[99,100.15],icp,max_time_gap=.11)


if __name__ == '__main__': unittest.main()
