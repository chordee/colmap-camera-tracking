import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colmap_to_3de import SceneLoadError, load_scene


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _write_points_txt(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: %d\n" % len(lines))
        for line in lines:
            f.write(line + "\n")


IDENTITY_CAMERA_JSON = {
    "fl_x": 1000.0, "fl_y": 1000.0, "cx": 960.0, "cy": 540.0,
    "w": 1920.0, "h": 1080.0, "sensor_w": 1920.0, "sensor_h": 1080.0,
    "frames": [
        {"file_path": "images_undistorted/frame_000002.jpg",
         "transform_matrix": [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        {"file_path": "images_undistorted/frame_000001.jpg",
         "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
    ],
}


class TestLoadScene(unittest.TestCase):
    def _make_scene(self, tmp_path, camera_json=None, point_lines=None):
        scene_dir = tmp_path / "scene"
        _write_json(scene_dir / "undistort" / "transforms_undistorted.json",
                     camera_json if camera_json is not None else IDENTITY_CAMERA_JSON)
        _write_points_txt(scene_dir / "sparse" / "0" / "points3D.txt",
                            point_lines if point_lines is not None else
                            ["1 1.0 2.0 3.0 255 0 0 0.5 1 0", "2 4.0 5.0 6.0 0 255 0 0.3 1 0"])
        return scene_dir

    def test_loads_intrinsics(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(Path(tmp))
            data = load_scene(str(scene_dir))
            self.assertEqual(data["fl_x"], 1000.0)
            self.assertEqual(data["w"], 1920.0)

    def test_frames_sorted_by_frame_number(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(Path(tmp))
            data = load_scene(str(scene_dir))
            self.assertEqual([f["frame_num"] for f in data["frames"]], [1, 2])

    def test_loads_points(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(Path(tmp))
            data = load_scene(str(scene_dir))
            self.assertEqual(len(data["points"]), 2)
            self.assertEqual(data["points"][0], {"id": "1", "x": 1.0, "y": 2.0, "z": 3.0})
            self.assertEqual(data["points"][1], {"id": "2", "x": 4.0, "y": 5.0, "z": 6.0})

    def test_missing_transforms_json_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene"
            _write_points_txt(scene_dir / "sparse" / "0" / "points3D.txt", ["1 1.0 2.0 3.0 255 0 0 0.5 1 0"])
            with self.assertRaises(SceneLoadError):
                load_scene(str(scene_dir))

    def test_missing_points3d_txt_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene"
            _write_json(scene_dir / "undistort" / "transforms_undistorted.json", IDENTITY_CAMERA_JSON)
            with self.assertRaises(SceneLoadError):
                load_scene(str(scene_dir))

    def test_multi_camera_json_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            multi_cam_json = {
                "frames": [
                    {"file_path": "images_undistorted/frame_000001.jpg", "fl_x": 1000.0, "fl_y": 1000.0,
                     "cx": 960.0, "cy": 540.0, "w": 1920.0, "h": 1080.0,
                     "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
                ],
            }
            scene_dir = self._make_scene(Path(tmp), camera_json=multi_cam_json)
            with self.assertRaises(SceneLoadError):
                load_scene(str(scene_dir))


if __name__ == "__main__":
    unittest.main()
