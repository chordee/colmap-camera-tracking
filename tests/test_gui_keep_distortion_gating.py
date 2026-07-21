import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui_autotracker import _keep_distortion_allowed


class TestKeepDistortionAllowed(unittest.TestCase):
    def test_opencv_allowed(self):
        self.assertTrue(_keep_distortion_allowed("OPENCV"))

    def test_full_opencv_allowed(self):
        self.assertTrue(_keep_distortion_allowed("FULL_OPENCV"))

    def test_fisheye_not_allowed(self):
        self.assertFalse(_keep_distortion_allowed("OPENCV_FISHEYE"))

    def test_auto_not_allowed(self):
        self.assertFalse(_keep_distortion_allowed("(auto)"))

    def test_simple_radial_not_allowed(self):
        self.assertFalse(_keep_distortion_allowed("SIMPLE_RADIAL"))


if __name__ == "__main__":
    unittest.main()
