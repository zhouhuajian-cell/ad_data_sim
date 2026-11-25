import unittest
import cv2
import numpy as np
from gnd_erase_refine import refine_local, ground_homography, refine_temporal

class EraseTest(unittest.TestCase):
    def test_refinement_remains_in_road_and_near_seed(self):
        im=np.full((80,100,3),80,np.uint8);road=np.ones((80,100),np.uint8);road[:,50:]=0
        mask=np.zeros((80,100),np.uint8);mask[40,45]=255
        im[40,46:49]=(40,190,230);im[40,65:75]=(40,190,230)
        out,info=refine_local(im,road,mask)
        self.assertGreater(info['added_pixels'],0)
        self.assertEqual(np.count_nonzero(out[:,50:]),0)
        self.assertEqual(out[40,70],0)
    def test_homography_and_unconfirmed_region(self):
        K=np.array([[100.,0,100],[0,100,100],[0,0,1]])
        tcw=np.array([[0,1,0,0],[0,0,-1,1.5],[1,0,0,0],[0,0,0,1.]])
        H=ground_homography(K,np.linalg.inv(tcw),np.eye(4),np.eye(4),0)
        self.assertIsNotNone(H)
        H=H/H[2,2];self.assertTrue(np.allclose(H,np.eye(3),atol=1e-5))
        current=np.zeros((30,30),np.uint8);previous=current.copy();previous[10,10]=255
        out,n=refine_temporal(current,previous,H,np.ones_like(current))
        self.assertEqual(n,0);self.assertEqual(out[10,10],0)
        current[10,11]=255;out,n=refine_temporal(current,previous,H,np.ones_like(current))
        self.assertEqual(n,1);self.assertEqual(out[10,10],255)
if __name__=='__main__': unittest.main()
