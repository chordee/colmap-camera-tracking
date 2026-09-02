from colmap_2d_tracks import is_exportable


def build_production_candidate_pool(tracks):
    """Production Candidate = structurally_valid AND non-conflict AND not
    EXACT_DUPLICATE (AGENTS_MASTER.md section 12) -- identical to
    is_exportable()'s predicate from Issue 02. Named separately so Issue
    04+ selection logic has one stable, documented entry point, and so a
    future change to export eligibility vs. candidate-pool eligibility can
    diverge without an implicit dependency between the two names."""
    return [t for t in tracks if is_exportable(t)]


def analyze_track_quality(track):
    """Per-track descriptive metrics (AGENTS_MASTER.md section 11).
    Analysis/diagnostic only -- never used here to exclude a track from
    the candidate pool or from export."""
    frames = sorted(obs["production_frame"] for obs in track["observations"])
    first_frame = frames[0]
    last_frame = frames[-1]
    temporal_span = last_frame - first_frame + 1
    observation_count = len(frames)

    gap_count = 0
    max_gap = 0
    for prev, cur in zip(frames, frames[1:]):
        gap = cur - prev - 1
        if gap > 0:
            gap_count += 1
            max_gap = max(max_gap, gap)

    return {
        "track_id": track["track_id"],
        "observation_count": observation_count,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "temporal_span": temporal_span,
        "actual_observation_coverage": observation_count / temporal_span,
        "gap_count": gap_count,
        "max_gap": max_gap,
    }
