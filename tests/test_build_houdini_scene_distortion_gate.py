import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_houdini_scene import _DISTORTION_KEYS, _has_meaningful_distortion


class TestHasMeaningfulDistortion(unittest.TestCase):
    def test_rectified_scene_present_but_zeroed_is_false(self):
        # Shape of a rectified (non --keep-distortion) transforms.json: all
        # distortion keys present but zeroed out. Must NOT be treated as
        # meaningful, or every ordinary scene would get a spurious
        # "OpenCV Distortion" spare-parameter folder.
        data = {"camera_model": "OPENCV"}
        data.update({key: 0.0 for key in _DISTORTION_KEYS})
        self.assertFalse(_has_meaningful_distortion(data.get("camera_model"), data))

    def test_keep_distortion_scene_with_one_nonzero_key_is_true(self):
        data = {"camera_model": "OPENCV"}
        data.update({key: 0.0 for key in _DISTORTION_KEYS})
        data["k1"] = 0.1
        self.assertTrue(_has_meaningful_distortion(data.get("camera_model"), data))

    def test_missing_camera_model_is_false(self):
        data = {key: 0.1 for key in _DISTORTION_KEYS}
        self.assertFalse(_has_meaningful_distortion(None, data))

    def test_none_camera_model_with_nonzero_keys_is_false(self):
        data = {"camera_model": None}
        data.update({key: 0.1 for key in _DISTORTION_KEYS})
        self.assertFalse(_has_meaningful_distortion(data.get("camera_model"), data))

    def test_empty_data_is_false(self):
        self.assertFalse(_has_meaningful_distortion(None, {}))

    def test_older_json_missing_distortion_fields_is_false(self):
        # JSON produced before distortion fields existed: camera_model may
        # or may not be present, but the keys are entirely absent.
        data = {"camera_model": "OPENCV"}
        self.assertFalse(_has_meaningful_distortion(data.get("camera_model"), data))


if __name__ == "__main__":
    unittest.main()
