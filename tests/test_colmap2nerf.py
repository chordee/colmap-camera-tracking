import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colmap2nerf import parse_camera_line


class TestParseCameraLine(unittest.TestCase):
    def test_opencv_line(self):
        line = "1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224"
        camera = parse_camera_line(line.split(" "))
        self.assertEqual(camera["camera_model"], "OPENCV")
        self.assertEqual(camera["w"], 3840.0)
        self.assertEqual(camera["h"], 2160.0)
        self.assertEqual(camera["fl_x"], 3178.27)
        self.assertEqual(camera["fl_y"], 3182.09)
        self.assertEqual(camera["cx"], 1920.0)
        self.assertEqual(camera["cy"], 1080.0)
        self.assertEqual(camera["k1"], 0.159668)
        self.assertEqual(camera["k2"], -0.231286)
        self.assertEqual(camera["p1"], -0.00123982)
        self.assertEqual(camera["p2"], 0.00272224)
        self.assertEqual(camera["k3"], 0)
        self.assertEqual(camera["k5"], 0)
        self.assertEqual(camera["k6"], 0)
        self.assertFalse(camera["is_fisheye"])

    def test_full_opencv_line(self):
        line = ("1 FULL_OPENCV 3840 2160 3178.27 3182.09 1920 1080 "
                "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8")
        camera = parse_camera_line(line.split(" "))
        self.assertEqual(camera["camera_model"], "FULL_OPENCV")
        self.assertEqual(camera["fl_x"], 3178.27)
        self.assertEqual(camera["fl_y"], 3182.09)
        self.assertEqual(camera["cx"], 1920.0)
        self.assertEqual(camera["cy"], 1080.0)
        self.assertEqual(camera["k1"], 0.1)
        self.assertEqual(camera["k2"], 0.2)
        self.assertEqual(camera["p1"], 0.3)
        self.assertEqual(camera["p2"], 0.4)
        self.assertEqual(camera["k3"], 0.5)
        self.assertEqual(camera["k4"], 0.6)
        self.assertEqual(camera["k5"], 0.7)
        self.assertEqual(camera["k6"], 0.8)
        self.assertFalse(camera["is_fisheye"])

    def test_full_opencv_fov_matches_formula(self):
        line = ("1 FULL_OPENCV 2000 1000 1000 1000 1000 500 "
                "0 0 0 0 0 0 0 0")
        camera = parse_camera_line(line.split(" "))
        expected_angle_x = math.atan(2000 / (1000 * 2)) * 2
        self.assertAlmostEqual(camera["camera_angle_x"], expected_angle_x)

    def test_opencv_fisheye_still_flags_is_fisheye(self):
        line = "1 OPENCV_FISHEYE 1920 1080 1000 1000 960 540 0.01 0.02 0.03 0.04"
        camera = parse_camera_line(line.split(" "))
        self.assertTrue(camera["is_fisheye"])
        self.assertEqual(camera["camera_model"], "OPENCV_FISHEYE")

    def test_unknown_model_falls_back_to_defaults(self):
        line = "1 SOME_FUTURE_MODEL 1920 1080 1000 1000"
        camera = parse_camera_line(line.split(" "))
        self.assertEqual(camera["camera_model"], "SOME_FUTURE_MODEL")
        self.assertEqual(camera["k1"], 0)
        self.assertEqual(camera["cx"], 960.0)
        self.assertEqual(camera["cy"], 540.0)


if __name__ == "__main__":
    unittest.main()
