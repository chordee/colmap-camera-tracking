import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from candidate_pool import build_production_candidate_pool, analyze_track_quality, build_temporal_coverage_audit


def _track(track_id, observations, structural_status="VALID", duplicate_status="UNIQUE", duplicate_of=None):
    return {
        "track_id": track_id, "track_name": track_id.split("::")[1],
        "observations": observations,
        "structural_status": structural_status,
        "duplicate_status": duplicate_status,
        "duplicate_of": duplicate_of,
    }


class TestBuildProductionCandidatePool(unittest.TestCase):
    def test_matches_is_exportable_predicate(self):
        obs = [{"production_frame": 1, "x": 1.0, "y": 1.0, "image_name": "a.jpg"}]
        tracks = [
            _track("colmap::1", obs, structural_status="VALID", duplicate_status="UNIQUE"),
            _track("colmap::2", obs, structural_status="CONFLICT"),
            _track("colmap::3", obs, structural_status="INVALID_COORDS"),
            _track("colmap::4", obs, structural_status="VALID", duplicate_status="EXACT_DUPLICATE", duplicate_of="colmap::1"),
        ]
        pool = build_production_candidate_pool(tracks)
        self.assertEqual([t["track_id"] for t in pool], ["colmap::1"])

    def test_empty_input_returns_empty(self):
        self.assertEqual(build_production_candidate_pool([]), [])


class TestAnalyzeTrackQuality(unittest.TestCase):
    def test_single_observation_track(self):
        track = _track("colmap::1", [
            {"production_frame": 5, "x": 1.0, "y": 1.0, "image_name": "a.jpg"},
        ])
        result = analyze_track_quality(track)
        self.assertEqual(result["track_id"], "colmap::1")
        self.assertEqual(result["observation_count"], 1)
        self.assertEqual(result["first_frame"], 5)
        self.assertEqual(result["last_frame"], 5)
        self.assertEqual(result["temporal_span"], 1)
        self.assertEqual(result["actual_observation_coverage"], 1.0)
        self.assertEqual(result["gap_count"], 0)
        self.assertEqual(result["max_gap"], 0)

    def test_no_gaps_contiguous_track(self):
        track = _track("colmap::1", [
            {"production_frame": f, "x": 1.0, "y": 1.0, "image_name": "a.jpg"} for f in (1, 2, 3, 4)
        ])
        result = analyze_track_quality(track)
        self.assertEqual(result["temporal_span"], 4)
        self.assertEqual(result["actual_observation_coverage"], 1.0)
        self.assertEqual(result["gap_count"], 0)
        self.assertEqual(result["max_gap"], 0)

    def test_single_gap(self):
        track = _track("colmap::1", [
            {"production_frame": f, "x": 1.0, "y": 1.0, "image_name": "a.jpg"} for f in (1, 2, 6, 7)
        ])
        result = analyze_track_quality(track)
        self.assertEqual(result["temporal_span"], 7)
        self.assertEqual(result["observation_count"], 4)
        self.assertAlmostEqual(result["actual_observation_coverage"], 4 / 7)
        self.assertEqual(result["gap_count"], 1)
        self.assertEqual(result["max_gap"], 3)

    def test_multiple_gaps_max_gap_picks_largest_not_last(self):
        # gaps between: 1->5 (gap 3), 5->7 (gap 1), 7->20 (gap 12)
        track = _track("colmap::1", [
            {"production_frame": f, "x": 1.0, "y": 1.0, "image_name": "a.jpg"} for f in (1, 5, 7, 20)
        ])
        result = analyze_track_quality(track)
        self.assertEqual(result["gap_count"], 3)
        self.assertEqual(result["max_gap"], 12)

    def test_unordered_observations_sorted_before_analysis(self):
        track = _track("colmap::1", [
            {"production_frame": f, "x": 1.0, "y": 1.0, "image_name": "a.jpg"} for f in (7, 1, 4)
        ])
        result = analyze_track_quality(track)
        self.assertEqual(result["first_frame"], 1)
        self.assertEqual(result["last_frame"], 7)


def _pool_track(track_id, frames):
    return _track(track_id, [
        {"production_frame": f, "x": 1.0, "y": 1.0, "image_name": "a.jpg"} for f in frames
    ])


class TestBuildTemporalCoverageAudit(unittest.TestCase):
    def test_frame_range_matches_observed_min_max(self):
        pool = [_pool_track("colmap::1", [3, 4, 5])]
        result = build_temporal_coverage_audit(pool)
        self.assertEqual(result["frame_min"], 3)
        self.assertEqual(result["frame_max"], 5)
        self.assertEqual(result["active_candidate_tracks"], {"3": 1, "4": 1, "5": 1})

    def test_zero_active_frames_included_in_range(self):
        # two tracks covering frames 1 and 3, frame 2 has no candidate at all
        pool = [_pool_track("colmap::1", [1]), _pool_track("colmap::2", [3])]
        result = build_temporal_coverage_audit(pool)
        self.assertEqual(result["active_candidate_tracks"], {"1": 1, "2": 0, "3": 1})
        self.assertEqual(result["frames_with_0_active"], 1)

    def test_bucket_boundaries(self):
        # frame 1: exactly 2 active -> 1_2 bucket
        # frame 2: exactly 3 active -> 3_5 bucket
        # frame 3: exactly 7 active -> 6_7 bucket
        # frame 4: exactly 8 active -> 8_plus bucket
        pool = []
        for i in range(2):
            pool.append(_pool_track(f"colmap::a{i}", [1]))
        for i in range(3):
            pool.append(_pool_track(f"colmap::b{i}", [2]))
        for i in range(7):
            pool.append(_pool_track(f"colmap::c{i}", [3]))
        for i in range(8):
            pool.append(_pool_track(f"colmap::d{i}", [4]))
        result = build_temporal_coverage_audit(pool)
        self.assertEqual(result["frames_with_1_2"], 1)
        self.assertEqual(result["frames_with_3_5"], 1)
        self.assertEqual(result["frames_with_6_7"], 1)
        self.assertEqual(result["frames_with_8_plus"], 1)
        self.assertEqual(result["frames_with_0_active"], 0)

    def test_min_median_max_simultaneous(self):
        pool = [_pool_track("colmap::1", [1, 2]), _pool_track("colmap::2", [2])]
        # frame 1: 1 active, frame 2: 2 active
        result = build_temporal_coverage_audit(pool)
        self.assertEqual(result["min_simultaneous"], 1)
        self.assertEqual(result["max_simultaneous"], 2)
        self.assertEqual(result["median_simultaneous"], 1.5)

    def test_longest_zero_track_range(self):
        # frames 1 and 10 have a candidate; 2-9 (8 frames) have none
        pool = [_pool_track("colmap::1", [1]), _pool_track("colmap::2", [10])]
        result = build_temporal_coverage_audit(pool)
        self.assertEqual(result["longest_zero_track_range"],
                          {"start_frame": 2, "end_frame": 9, "length": 8})

    def test_longest_low_support_range_threshold_is_2(self):
        # frame 1: 1 active, frame 2: 2 active, frame 3: 2 active, frame 4: 3 active (not low)
        pool = [_pool_track("colmap::a", [1]),
                _pool_track("colmap::b", [2]), _pool_track("colmap::c", [2]),
                _pool_track("colmap::d", [3]), _pool_track("colmap::e", [3]),
                _pool_track("colmap::f", [4]), _pool_track("colmap::g", [4]), _pool_track("colmap::h", [4])]
        result = build_temporal_coverage_audit(pool)
        # frames 1(1),2(2),3(2),4(3): low-support (<=2) run is frames 1-3, length 3
        self.assertEqual(result["longest_low_support_range"],
                          {"start_frame": 1, "end_frame": 3, "length": 3})

    def test_boundary_window_shorter_than_range_does_not_crash(self):
        # range of only 3 frames, boundary_window default 10
        pool = [_pool_track("colmap::1", [1, 2, 3])]
        result = build_temporal_coverage_audit(pool, boundary_window=10)
        self.assertEqual(result["opening_boundary_minimum"], 1)
        self.assertEqual(result["ending_boundary_minimum"], 1)

    def test_opening_and_ending_boundary_minimum(self):
        # base track covers frames 1..20 with exactly 1 active each; two extra
        # observations bump frame 3 and frame 20 to 2 active, but the window
        # minimum must still be 1 (from the other frames in each window).
        pool = [_pool_track("colmap::1", list(range(1, 21)))]
        pool.append(_pool_track("colmap::2", [3]))
        pool.append(_pool_track("colmap::3", [20]))
        result = build_temporal_coverage_audit(pool, boundary_window=5)
        self.assertEqual(result["opening_boundary_minimum"], 1)
        self.assertEqual(result["ending_boundary_minimum"], 1)

    def test_empty_pool_returns_empty_range(self):
        result = build_temporal_coverage_audit([])
        self.assertIsNone(result["frame_min"])
        self.assertIsNone(result["frame_max"])
        self.assertEqual(result["active_candidate_tracks"], {})
        self.assertEqual(result["min_simultaneous"], 0)
        self.assertEqual(result["max_simultaneous"], 0)
        self.assertEqual(result["median_simultaneous"], 0)


if __name__ == "__main__":
    unittest.main()
