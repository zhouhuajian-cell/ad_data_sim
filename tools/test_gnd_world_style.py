import unittest
import cv2
import numpy as np
from gnd_world_style import prepare_texture, composite


class StyleTest(unittest.TestCase):
    def test_wear_dirt_and_stability(self):
        tex = np.full((128, 32, 4), 255, np.uint8)
        clean = prepare_texture(tex, 0, 0, 'a')
        worn = prepare_texture(tex, .8, .6, 'a')
        self.assertTrue(np.array_equal(worn, prepare_texture(tex, .8, .6, 'a')))
        self.assertLess(float(worn[:,:,3].mean()), float(clean[:,:,3].mean()))
        self.assertLess(float(worn[:,:,:3].mean()), float(clean[:,:,:3].mean()))

    def test_opacity_reflection_blur_and_road_gate(self):
        bg=np.full((40,40,3), 80, np.uint8)
        rgb=np.full((40,40,3), 180, np.uint8)
        alpha=np.zeros((40,40),np.uint8);alpha[10:30,10:30]=255
        road=np.ones((40,40),np.uint8);road[:,20:]=0
        low,_=composite(bg,rgb,alpha,road,opacity=.2,reflectivity=1.2,optical_blur=.5,occlusion=True)
        high,a=composite(bg,rgb,alpha,road,opacity=.9,reflectivity=3,optical_blur=1.2,occlusion=True)
        self.assertGreater(int(high[20,15,0]),int(low[20,15,0]))
        self.assertEqual(int(a[20,25]),0)
        opened,_=composite(bg,rgb,alpha,road,opacity=.9,reflectivity=3,optical_blur=1.2,occlusion=False)
        self.assertGreater(int(opened[20,25,0]),int(bg[20,25,0]))

if __name__ == '__main__': unittest.main()
