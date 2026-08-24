import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from colmap_to_3de import SceneLoadError, load_scene, colmap_matrix_to_3de, flip_point_to_3de, points_to_survey_lines, write_survey_points_txt, write_camera_import_script, sample_points, filter_by_min_track_length


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
            self.assertEqual(data["points"][0], {"id": "1", "x": 1.0, "y": 2.0, "z": 3.0, "track_length": 1})
            self.assertEqual(data["points"][1], {"id": "2", "x": 4.0, "y": 5.0, "z": 6.0, "track_length": 1})

    def test_loads_track_length_from_track_array(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # POINT3D_ID X Y Z R G B ERROR TRACK[](IMAGE_ID,POINT2D_IDX)... --
            # 3 track pairs (1,0) (2,0) (3,0) -> track_length 3.
            scene_dir = self._make_scene(Path(tmp), point_lines=[
                "1 1.0 2.0 3.0 255 0 0 0.5 1 0 2 0 3 0",
            ])
            data = load_scene(str(scene_dir))
            self.assertEqual(data["points"][0]["track_length"], 3)

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


class TestColmapMatrixTo3de(unittest.TestCase):
    def test_identity_frame(self):
        matrix = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        position, rotation = colmap_matrix_to_3de(matrix)
        self.assertEqual(position, [0.0, 0.0, 0.0])
        self.assertEqual(rotation, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_frame1_matches_real_houdini_bake(self):
        # Real transform_matrix from demo-test/walking-forest-exr-output's frame 1.
        matrix = [
            [1.0, 4.286811358594259e-16, -6.514761575350366e-16, 0.24618224016037848],
            [4.2868113585942726e-16, -1.0, 2.0864680150277905e-15, -0.3791985651695287],
            [-6.514761575350358e-16, -2.0864680150277913e-15, -1.0, -6.332398622318554],
            [0.0, 0.0, 0.0, 1.0],
        ]
        position, rotation = colmap_matrix_to_3de(matrix)
        # Verified against hython: cam.parm('tx'/'ty'/'tz').evalAsFloatAtFrame(1)
        # on the real walking-forest-exr.hip.
        self.assertAlmostEqual(position[0], 0.24618224016037848, places=9)
        self.assertAlmostEqual(position[1], 0.3791985651695287, places=9)
        self.assertAlmostEqual(position[2], 6.332398622318554, places=9)
        # Near-identity rotation (frame 1 looks straight down -Z, matching the
        # source matrix's near-diag(1,-1,-1) camera-local basis).
        self.assertAlmostEqual(rotation[0][0], 1.0, places=6)
        self.assertAlmostEqual(rotation[1][1], 1.0, places=6)
        self.assertAlmostEqual(rotation[2][2], 1.0, places=6)

    def test_frame2_matches_real_houdini_bake(self):
        # Real transform_matrix from demo-test/walking-forest-exr-output's frame 2.
        matrix = [
            [0.9999997238518143, 6.481495497135973e-05, 0.00074033459788004, 0.24626607526936112],
            [6.544238430871263e-05, -0.9999996387291866, -0.0008475014987890106, -0.3841761334195393],
            [0.0007402793996472756, 0.0008475497140142808, -0.9999993668227459, -6.2811209126363075],
            [0.0, 0.0, 0.0, 1.0],
        ]
        position, rotation = colmap_matrix_to_3de(matrix)
        self.assertAlmostEqual(position[0], 0.24626607526936112, places=9)
        self.assertAlmostEqual(position[1], 0.3841761334195393, places=9)
        self.assertAlmostEqual(position[2], 6.2811209126363075, places=9)
        # rotation = flip @ M[:3,:3] -- verified by hand against the transpose
        # of hython's extractRotates()-reconstructed matrix for this frame.
        expected_rotation = [
            [0.9999997238518143, 6.481495497135973e-05, 0.00074033459788004],
            [-6.544238430871263e-05, 0.9999996387291866, 0.0008475014987890106],
            [-0.0007402793996472756, -0.0008475497140142808, 0.9999993668227459],
        ]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(rotation[i][j], expected_rotation[i][j], places=9)


class TestFlipPointTo3de(unittest.TestCase):
    def test_flips_y_and_z(self):
        self.assertEqual(flip_point_to_3de(1.0, 2.0, 3.0), (1.0, -2.0, -3.0))


class TestPointsToSurveyLines(unittest.TestCase):
    def test_flips_and_names_points(self):
        points = [{"id": "42", "x": 1.0, "y": 2.0, "z": 3.0}]
        lines = points_to_survey_lines(points)
        self.assertEqual(lines, [("p42", 1.0, -2.0, -3.0)])

    def test_preserves_point_order(self):
        points = [{"id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": "2", "x": 1.0, "y": 1.0, "z": 1.0}]
        lines = points_to_survey_lines(points)
        self.assertEqual([name for name, x, y, z in lines], ["p1", "p2"])


class TestWriteSurveyPointsTxt(unittest.TestCase):
    def test_writes_four_column_format(self):
        import tempfile
        points = [{"id": "1", "x": 1.5, "y": 2.5, "z": 3.5}]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "points.txt")
            write_survey_points_txt(points, out_path)
            with open(out_path) as f:
                lines = f.read().splitlines()
        self.assertEqual(len(lines), 1)
        parts = lines[0].split()
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], "p1")
        self.assertAlmostEqual(float(parts[1]), 1.5)
        self.assertAlmostEqual(float(parts[2]), -2.5)
        self.assertAlmostEqual(float(parts[3]), -3.5)

    def test_writes_one_line_per_point(self):
        import tempfile
        points = [{"id": "1", "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": "2", "x": 1.0, "y": 1.0, "z": 1.0},
                  {"id": "3", "x": 2.0, "y": 2.0, "z": 2.0}]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "points.txt")
            write_survey_points_txt(points, out_path)
            with open(out_path) as f:
                lines = f.read().splitlines()
        self.assertEqual(len(lines), 3)


class TestWriteCameraImportScript(unittest.TestCase):
    def test_generates_script_with_3de4_menu_header_and_embedded_data(self):
        import tempfile
        scene = {
            "w": 1920.0, "h": 1080.0, "sensor_w": 1920.0, "sensor_h": 1080.0,
            "fl_x": 960.0, "fl_y": 960.0, "cx": 960.0, "cy": 540.0,
            "frames": [
                {"frame_num": 1, "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
                {"frame_num": 2, "transform_matrix": [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            ],
            "points": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "import_camera.py")
            write_camera_import_script(scene, "demo_scene", sensor_width_mm=36.0, out_path=out_path,
                                        images_dir="/fake/scene/undistort/images_undistorted")
            with open(out_path) as f:
                content = f.read()

        # 3DE Script Database discovers scripts via these header comments --
        # matches the convention used by every official script in
        # D:\Programs\3DE4_win64_r5\sys_data\py_scripts\ (e.g. export_houdini.py).
        self.assertIn("# 3DE4.script.name:", content)
        self.assertIn("# 3DE4.script.gui:\tMain Window::3DE4::File::Import", content)
        self.assertIn("import tde4", content)
        self.assertIn("tde4.createPGroup(\"CAMERA\")", content)
        self.assertIn("tde4.createCamera(\"SEQUENCE\")", content)
        self.assertIn("tde4.createLens()", content)
        # frame count and image dimensions must be embedded literally
        self.assertIn("tde4.setCameraSequenceAttr(camera_id, 1, 2, 1)", content)
        self.assertIn("tde4.setCameraImageWidth(camera_id, 1920)", content)
        self.assertIn("tde4.setCameraImageHeight(camera_id, 1080)", content)
        self.assertIn("tde4.setCameraPath(camera_id, \"/fake/scene/undistort/images_undistorted/frame_######.jpg\")", content)
        # per-frame position/rotation must be present for both frames
        self.assertIn("tde4.setPGroupPosition3D(pgroup_id, camera_id, 1,", content)
        self.assertIn("tde4.setPGroupPosition3D(pgroup_id, camera_id, 2,", content)
        self.assertIn("tde4.setPGroupRotation3D(pgroup_id, camera_id, 1,", content)
        # focal length: fl_x=960px, sensor_w=1920px, sensor_width_mm=36.0
        # -> focal_mm = (960/1920)*36.0 = 18.0mm -> 1.8cm
        self.assertIn("tde4.setCameraFocalLength(camera_id, 1, 1.8)", content)
        self.assertIn("tde4.setCameraFocalLength(camera_id, 2, 1.8)", content)
        # film back: fback_w_mm = 36.0*(1920/1920) = 36.0mm -> 3.6cm;
        # fback_h_mm = 36.0*(1080/1920) = 20.25mm -> 2.025cm
        self.assertIn("tde4.setLensFBackWidth(lens_id, 3.6)", content)
        self.assertIn("tde4.setLensFBackHeight(lens_id, 2.025)", content)
        # principal point centered (cx=w/2, cy=h/2) -> zero lens center offset
        self.assertIn("tde4.setLensLensCenterX(lens_id, 0.0)", content)
        self.assertIn("tde4.setLensLensCenterY(lens_id, 0.0)", content)
        # newly created pgroup/camera must be made current so 3DE's UI
        # (timeline, viewer, Import Survey Textfile) targets them
        self.assertIn("tde4.setCurrentPGroup(pgroup_id)", content)
        self.assertIn("tde4.setCurrentCamera(camera_id)", content)


class TestFilterByMinTrackLength(unittest.TestCase):
    def _points_with_tracks(self, lengths):
        return [{"id": str(i), "x": 0.0, "y": 0.0, "z": 0.0, "track_length": t}
                 for i, t in enumerate(lengths)]

    def test_no_threshold_returns_all_points_unchanged(self):
        points = self._points_with_tracks([1, 2, 3])
        self.assertEqual(filter_by_min_track_length(points, None), points)
        self.assertEqual(filter_by_min_track_length(points, 0), points)

    def test_drops_points_below_threshold(self):
        points = self._points_with_tracks([1, 2, 3, 5, 10])
        result = filter_by_min_track_length(points, 3)
        self.assertEqual([p["track_length"] for p in result], [3, 5, 10])

    def test_keeps_points_exactly_at_threshold(self):
        points = self._points_with_tracks([2, 3])
        result = filter_by_min_track_length(points, 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["track_length"], 3)

    def test_threshold_above_all_points_returns_empty(self):
        points = self._points_with_tracks([1, 2, 3])
        result = filter_by_min_track_length(points, 100)
        self.assertEqual(result, [])


class TestSamplePoints(unittest.TestCase):
    def _points(self, n):
        return [{"id": str(i), "x": float(i), "y": 0.0, "z": 0.0} for i in range(n)]

    def test_no_limit_returns_all_points_unchanged(self):
        points = self._points(10)
        self.assertEqual(sample_points(points, None), points)
        self.assertEqual(sample_points(points, 0), points)

    def test_limit_returns_exact_count(self):
        points = self._points(1000)
        result = sample_points(points, 100, seed=42)
        self.assertEqual(len(result), 100)

    def test_limit_greater_than_available_returns_all(self):
        points = self._points(10)
        result = sample_points(points, 100, seed=42)
        self.assertEqual(len(result), 10)

    def test_sample_is_a_subset_with_no_duplicates(self):
        points = self._points(1000)
        result = sample_points(points, 100, seed=42)
        ids = [p["id"] for p in result]
        self.assertEqual(len(ids), len(set(ids)))
        for p in result:
            self.assertIn(p, points)

    def test_seed_makes_sampling_deterministic(self):
        points = self._points(1000)
        result1 = sample_points(points, 50, seed=7)
        result2 = sample_points(points, 50, seed=7)
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
