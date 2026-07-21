import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_autotracker import undistort_output_paths, validate_keep_distortion_args


class TestValidateKeepDistortionArgs(unittest.TestCase):
    def test_opencv_with_flag_is_valid(self):
        self.assertIsNone(validate_keep_distortion_args("OPENCV", True))

    def test_full_opencv_with_flag_is_valid(self):
        self.assertIsNone(validate_keep_distortion_args("FULL_OPENCV", True))

    def test_lowercase_model_is_normalized(self):
        self.assertIsNone(validate_keep_distortion_args("opencv", True))

    def test_fisheye_with_flag_is_invalid(self):
        error = validate_keep_distortion_args("OPENCV_FISHEYE", True)
        self.assertIsNotNone(error)
        self.assertIn("OPENCV_FISHEYE", error)

    def test_flag_not_set_is_always_valid(self):
        self.assertIsNone(validate_keep_distortion_args("OPENCV_FISHEYE", False))
        self.assertIsNone(validate_keep_distortion_args("SIMPLE_RADIAL", False))


class TestUndistortOutputPaths(unittest.TestCase):
    def test_default_path(self):
        output_dir, json_filename = undistort_output_paths("/out/sceneA", False)
        self.assertEqual(output_dir, __import__("os").path.join("/out/sceneA", "undistort"))
        self.assertEqual(json_filename, "transforms_undistorted.json")

    def test_keep_distortion_path(self):
        output_dir, json_filename = undistort_output_paths("/out/sceneA", True)
        self.assertEqual(output_dir, __import__("os").path.join("/out/sceneA", "keep_distortion"))
        self.assertEqual(json_filename, "transforms_original.json")


if __name__ == "__main__":
    unittest.main()
