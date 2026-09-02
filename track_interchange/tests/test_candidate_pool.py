import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from candidate_pool import build_production_candidate_pool, analyze_track_quality


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


if __name__ == "__main__":
    unittest.main()
