import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from undistortionNerfstudioColmap import (
    build_keep_distortion_json,
    validate_keep_distortion_model,
)


class TestValidateKeepDistortionModel(unittest.TestCase):
    def test_opencv_is_allowed(self):
        validate_keep_distortion_model("OPENCV")  # must not raise

    def test_full_opencv_is_allowed(self):
        validate_keep_distortion_model("FULL_OPENCV")  # must not raise

    def test_fisheye_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_keep_distortion_model("OPENCV_FISHEYE")

    def test_simple_radial_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_keep_distortion_model("SIMPLE_RADIAL")

    def test_missing_camera_model_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_keep_distortion_model(None)


class TestBuildKeepDistortionJson(unittest.TestCase):
    def test_preserves_distortion_fields(self):
        data = {
            "camera_model": "OPENCV",
            "fl_x": 1000.0, "fl_y": 1000.0, "cx": 960.0, "cy": 540.0,
            "k1": 0.1, "k2": 0.2, "k3": 0.0, "k4": 0.0,
            "k5": 0.0, "k6": 0.0, "p1": 0.01, "p2": 0.02,
            "w": 1920.0, "h": 1080.0,
            "frames": [{"file_path": "./scene/images/frame_000001.jpg", "sharpness": 5.0}],
        }
        result = build_keep_distortion_json(data, json_dir="/root/output", output_dir="/root/output/scene/keep_distortion")
        self.assertEqual(result["k1"], 0.1)
        self.assertEqual(result["p2"], 0.02)
        self.assertEqual(result["camera_model"], "OPENCV")

    def test_rewrites_frame_paths_relative_to_output_dir(self):
        data = {
            "frames": [{"file_path": "./scene/images/frame_000001.jpg"}],
        }
        result = build_keep_distortion_json(data, json_dir="/root/output", output_dir="/root/output/scene/keep_distortion")
        # /root/output/scene/images/frame_000001.jpg, made relative to
        # /root/output/scene/keep_distortion, is ../images/frame_000001.jpg
        self.assertEqual(result["frames"][0]["file_path"], "../images/frame_000001.jpg")

    def test_does_not_mutate_input(self):
        data = {"frames": [{"file_path": "./scene/images/frame_000001.jpg"}]}
        build_keep_distortion_json(data, json_dir="/root/output", output_dir="/root/output/scene/keep_distortion")
        self.assertEqual(data["frames"][0]["file_path"], "./scene/images/frame_000001.jpg")


if __name__ == "__main__":
    unittest.main()
