import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colmap_2d_tracks import SceneLoadError, load_scene


CAMERAS_TXT = """# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 SIMPLE_RADIAL 1920 1080 1000.0 960 540 0.0
"""

IMAGES_TXT = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 2
1 1 0 0 0 0 0 0 1 frame_000001.jpg
100.0 200.0 5 300.0 400.0 -1 500.0 600.0 7
2 1 0 0 0 0 0 0 1 frame_000002.jpg
110.0 210.0 5 510.0 610.0 7
"""

# Second fixture: frame 1 has point 5 twice at different pixel positions --
# a same-track/same-frame conflict. Track 5 must be dropped entirely; track
# 9 (conflict-free) must still come through.
IMAGES_TXT_WITH_CONFLICT = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 1
1 1 0 0 0 0 0 0 1 frame_000001.jpg
100.0 200.0 5 150.0 250.0 5 300.0 400.0 9
"""


def _make_scene(tmp_path, images_txt=IMAGES_TXT, cameras_txt=CAMERAS_TXT):
    scene_dir = tmp_path / "scene"
    sparse_dir = scene_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "cameras.txt").write_text(cameras_txt)
    (sparse_dir / "images.txt").write_text(images_txt)
    return scene_dir


class TestLoadScene(unittest.TestCase):
    def test_loads_image_size(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = _make_scene(Path(tmp))
            data = load_scene(str(scene_dir))
            self.assertEqual(data["width"], 1920)
            self.assertEqual(data["height"], 1080)

    def test_builds_tracks_by_point3d_id(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = _make_scene(Path(tmp))
            data = load_scene(str(scene_dir))
            track_ids = sorted(t["track_id"] for t in data["tracks"])
            self.assertEqual(track_ids, ["colmap::5", "colmap::7"])

    def test_track_name_and_observations(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = _make_scene(Path(tmp))
            data = load_scene(str(scene_dir))
            track5 = next(t for t in data["tracks"] if t["track_id"] == "colmap::5")
            self.assertEqual(track5["track_name"], "p5")
            self.assertEqual(track5["observations"], [
                {"production_frame": 1, "x": 100.0, "y": 200.0},
                {"production_frame": 2, "x": 110.0, "y": 210.0},
            ])

    def test_excludes_point3d_id_negative_one(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = _make_scene(Path(tmp))
            data = load_scene(str(scene_dir))
            track_ids = [t["track_id"] for t in data["tracks"]]
            self.assertNotIn("colmap::-1", track_ids)

    def test_same_frame_conflict_drops_track_and_is_counted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = _make_scene(Path(tmp), images_txt=IMAGES_TXT_WITH_CONFLICT)
            data = load_scene(str(scene_dir))
            track_ids = [t["track_id"] for t in data["tracks"]]
            self.assertNotIn("colmap::5", track_ids)
            self.assertIn("colmap::9", track_ids)
            self.assertEqual(data["conflict_count"], 1)

    def test_missing_cameras_txt_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene"
            sparse_dir = scene_dir / "sparse" / "0"
            sparse_dir.mkdir(parents=True)
            (sparse_dir / "images.txt").write_text(IMAGES_TXT)
            with self.assertRaises(SceneLoadError):
                load_scene(str(scene_dir))

    def test_missing_images_txt_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene"
            sparse_dir = scene_dir / "sparse" / "0"
            sparse_dir.mkdir(parents=True)
            (sparse_dir / "cameras.txt").write_text(CAMERAS_TXT)
            with self.assertRaises(SceneLoadError):
                load_scene(str(scene_dir))


if __name__ == "__main__":
    unittest.main()
