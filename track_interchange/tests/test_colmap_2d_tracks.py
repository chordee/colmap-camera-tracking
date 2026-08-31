import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colmap_2d_tracks import SceneLoadError, load_scene, filter_by_min_observations, sample_tracks


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


def _tracks_with_observation_counts(counts):
    return [
        {"track_id": f"colmap::{i}", "track_name": f"p{i}",
         "observations": [{"production_frame": f, "x": 0.0, "y": 0.0} for f in range(n)]}
        for i, n in enumerate(counts)
    ]


class TestFilterByMinObservations(unittest.TestCase):
    def test_no_threshold_returns_all_tracks_unchanged(self):
        tracks = _tracks_with_observation_counts([1, 2, 3])
        self.assertEqual(filter_by_min_observations(tracks, None), tracks)
        self.assertEqual(filter_by_min_observations(tracks, 0), tracks)

    def test_drops_tracks_below_threshold(self):
        tracks = _tracks_with_observation_counts([1, 2, 3, 5, 10])
        result = filter_by_min_observations(tracks, 3)
        self.assertEqual([len(t["observations"]) for t in result], [3, 5, 10])

    def test_keeps_tracks_exactly_at_threshold(self):
        tracks = _tracks_with_observation_counts([2, 3])
        result = filter_by_min_observations(tracks, 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["observations"]), 3)


class TestSampleTracks(unittest.TestCase):
    def _tracks(self, n):
        return [{"track_id": f"colmap::{i}", "track_name": f"p{i}", "observations": []}
                 for i in range(n)]

    def test_no_limit_returns_all_tracks_unchanged(self):
        tracks = self._tracks(10)
        self.assertEqual(sample_tracks(tracks, None), tracks)
        self.assertEqual(sample_tracks(tracks, 0), tracks)

    def test_limit_returns_exact_count(self):
        tracks = self._tracks(1000)
        result = sample_tracks(tracks, 100, seed=42)
        self.assertEqual(len(result), 100)

    def test_limit_greater_than_available_returns_all(self):
        tracks = self._tracks(10)
        result = sample_tracks(tracks, 100, seed=42)
        self.assertEqual(len(result), 10)

    def test_sample_is_a_subset_with_no_duplicates(self):
        tracks = self._tracks(1000)
        result = sample_tracks(tracks, 100, seed=42)
        ids = [t["track_id"] for t in result]
        self.assertEqual(len(ids), len(set(ids)))
        for t in result:
            self.assertIn(t, tracks)


if __name__ == "__main__":
    unittest.main()
