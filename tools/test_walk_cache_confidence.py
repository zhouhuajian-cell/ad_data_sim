import os
import tempfile
import unittest
import walk_cache

class CacheTest(unittest.TestCase):
    def test_detector_and_threshold_are_part_of_key(self):
        with tempfile.TemporaryDirectory() as d:
            walk_cache.DB_PATH=os.path.join(d,'cache.db')
            image=os.path.join(d,'frame.jpg')
            with open(image,'wb') as f:f.write(b'frame')
            walk_cache.put_scan(image,[{'conf':.22}],conf=.2,detector='v1')
            self.assertEqual(walk_cache.get_scan(image,conf=.2,detector='v1'),[{'conf':.22}])
            self.assertIsNone(walk_cache.get_scan(image,conf=.15,detector='v1'))
            self.assertIsNone(walk_cache.get_scan(image,conf=.2,detector='v2'))
            walk_cache.put_scan(image,[{'conf':.16}],conf=.15,detector='v1')
            self.assertEqual(walk_cache.get_scan(image,conf=.15,detector='v1'),[{'conf':.16}])
if __name__=='__main__':unittest.main()
