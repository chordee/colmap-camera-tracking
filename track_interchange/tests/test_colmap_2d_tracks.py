import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colmap_2d_tracks import SceneLoadError, load_scene, filter_by_min_observations, sample_tracks, write_3de_2d_tracks_txt


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


class TestWrite3deTracksTxt(unittest.TestCase):
    def _tracks(self):
        return [
            {"track_id": "colmap::5", "track_name": "p5", "observations": [
                {"production_frame": 1, "x": 100.0, "y": 200.0},
                {"production_frame": 2, "x": 110.0, "y": 210.0},
            ]},
            {"track_id": "colmap::7", "track_name": "p7", "observations": [
                {"production_frame": 1, "x": 500.0, "y": 600.0},
            ]},
        ]

    def test_no_blank_lines_or_comments(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "tracks.txt")
            write_3de_2d_tracks_txt(self._tracks(), production_start_frame=1, out_path=out_path)
            with open(out_path) as f:
                content = f.read()
        self.assertNotIn("\n\n", content)
        self.assertNotIn("#", content)
        lines = content.rstrip("\n").split("\n")
        for line in lines:
            self.assertEqual(line, line.strip(), f"line has leading/trailing whitespace: {line!r}")

    def test_exact_structure(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "tracks.txt")
            write_3de_2d_tracks_txt(self._tracks(), production_start_frame=1, out_path=out_path)
            with open(out_path) as f:
                lines = f.read().rstrip("\n").split("\n")
        self.assertEqual(lines[0], "2")            # TRACK_COUNT
        self.assertEqual(lines[1], "p5")            # TRACK_NAME
        self.assertEqual(lines[2], "0")              # static field
        self.assertEqual(lines[3], "2")              # SAMPLE_COUNT for p5
        self.assertEqual(lines[4], "1 100.000000000000000 200.000000000000000")
        self.assertEqual(lines[5], "2 110.000000000000000 210.000000000000000")
        self.assertEqual(lines[6], "p7")
        self.assertEqual(lines[7], "0")
        self.assertEqual(lines[8], "1")              # SAMPLE_COUNT for p7
        self.assertEqual(lines[9], "1 500.000000000000000 600.000000000000000")
        self.assertEqual(len(lines), 10)

    def test_production_start_frame_offset(self):
        import tempfile
        tracks = [{"track_id": "colmap::5", "track_name": "p5", "observations": [
            {"production_frame": 1001, "x": 1.0, "y": 2.0},
            {"production_frame": 1005, "x": 3.0, "y": 4.0},
        ]}]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "tracks.txt")
            write_3de_2d_tracks_txt(tracks, production_start_frame=1001, out_path=out_path)
            with open(out_path) as f:
                lines = f.read().rstrip("\n").split("\n")
        # 3de_internal_frame = production_frame - production_start_frame + 1
        self.assertTrue(lines[4].startswith("1 "))   # 1001 - 1001 + 1 = 1
        self.assertTrue(lines[5].startswith("5 "))   # 1005 - 1001 + 1 = 5

    def test_natural_gaps_not_filled(self):
        import tempfile
        tracks = [{"track_id": "colmap::5", "track_name": "p5", "observations": [
            {"production_frame": 1, "x": 0.0, "y": 0.0},
            {"production_frame": 2, "x": 0.0, "y": 0.0},
            {"production_frame": 3, "x": 0.0, "y": 0.0},
            {"production_frame": 7, "x": 0.0, "y": 0.0},
            {"production_frame": 8, "x": 0.0, "y": 0.0},
        ]}]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "tracks.txt")
            write_3de_2d_tracks_txt(tracks, production_start_frame=1, out_path=out_path)
            with open(out_path) as f:
                lines = f.read().rstrip("\n").split("\n")
        # SAMPLE_COUNT must be 5 (actual rows), not 8 (frame span)
        self.assertEqual(lines[3], "5")
        self.assertEqual(len(lines), 4 + 5)

    def test_no_crlf_line_endings(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "tracks.txt")
            write_3de_2d_tracks_txt(self._tracks(), production_start_frame=1, out_path=out_path)
            with open(out_path, "rb") as f:
                data = f.read()
        self.assertNotIn(b"\r\n", data)

    def test_empty_track_list_writes_zero(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "tracks.txt")
            write_3de_2d_tracks_txt([], production_start_frame=1, out_path=out_path)
            with open(out_path) as f:
                content = f.read()
        self.assertEqual(content, "0\n")


if __name__ == "__main__":
    unittest.main()
