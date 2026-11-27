import unittest
from unittest import mock
from obj_track_projection import projected_bbox,too_much_overlap
import smooth_track


class ProjectionTest(unittest.TestCase):
    def test_planner_tries_another_corridor_if_image_boxes_overlap(self):
        stems=['1','2','3']
        def build(_poses,_stems,offset_m,lateral_m,**_kwargs):
            return {'lateral_m':lateral_m,'positions':{s:(offset_m,lateral_m,0.) for s in stems}}
        def convert(track,_poses,_stems):
            return {s:{'x':track['positions'][s][0],'y':track['positions'][s][1],'yaw_rad':0.} for s in stems}
        def image_clear(local,_size,accepted):
            return not accepted or local[stems[0]]['y']>0
        with mock.patch.object(smooth_track,'build_ego_path_track',side_effect=build), \
             mock.patch.object(smooth_track,'world_to_lidar',side_effect=convert), \
             mock.patch.object(smooth_track,'precheck_track',return_value={'ok':True,'total':3,'collision_frames':0}):
            result=smooth_track.plan_multiple_tracks({},stems,{}, {},
                [{'uid':'a','lwh':[4,2,1.5]},{'uid':'b','lwh':[4,2,1.5]}],
                offset_candidates=(18.,),lateral_candidates=(0.,-7.,7.),image_pair_clear=image_clear)
        self.assertEqual(result[1]['track']['lateral_m'],7.)
    def test_projected_vehicle_and_overlap(self):
        cam={'R':[[0,-1,0],[0,0,-1],[1,0,0]],'t':[0,1.5,0],
             'fx':500,'fy':500,'cx':320,'cy':240}
        local={'x':20.,'y':0.,'yaw_rad':0.}
        b=projected_bbox(local,[4.,2.,1.5],cam,(480,640),0.)
        self.assertIsNotNone(b)
        self.assertTrue(too_much_overlap(b,b))
        self.assertFalse(too_much_overlap(b,[0,0,5,5]))
        self.assertIsNone(projected_bbox({'x':-10.,'y':0.,'yaw_rad':0.},[4.,2.,1.5],cam,(480,640),0.))


if __name__=='__main__':unittest.main()
