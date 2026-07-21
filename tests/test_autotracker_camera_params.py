import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotracker import build_camera_params_str


class TestBuildCameraParamsStr(unittest.TestCase):
    def test_simple_pinhole_is_3_params(self):
        result = build_camera_params_str("SIMPLE_PINHOLE", 1000.0, 960.0, 540.0)
        self.assertEqual(result, "1000.0,960.0,540.0")

    def test_pinhole_is_4_params(self):
        result = build_camera_params_str("PINHOLE", 1000.0, 960.0, 540.0)
        self.assertEqual(result.split(","), ["1000.0", "1000.0", "960.0", "540.0"])

    def test_radial_is_5_params(self):
        result = build_camera_params_str("RADIAL", 1000.0, 960.0, 540.0)
        self.assertEqual(len(result.split(",")), 5)

    def test_opencv_is_8_params(self):
        result = build_camera_params_str("OPENCV", 1000.0, 960.0, 540.0)
        self.assertEqual(len(result.split(",")), 8)

    def test_full_opencv_is_12_params(self):
        result = build_camera_params_str("FULL_OPENCV", 1000.0, 960.0, 540.0)
        parts = result.split(",")
        self.assertEqual(len(parts), 12)
        self.assertEqual(parts[0], "1000.0")
        self.assertEqual(parts[1], "1000.0")
        self.assertEqual(parts[2], "960.0")
        self.assertEqual(parts[3], "540.0")

    def test_opencv_fisheye_is_8_params(self):
        result = build_camera_params_str("OPENCV_FISHEYE", 1000.0, 960.0, 540.0)
        self.assertEqual(len(result.split(",")), 8)


if __name__ == "__main__":
    unittest.main()
