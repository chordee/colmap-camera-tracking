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


def _longest_run(frame_range, active_by_frame, predicate):
    """Longest consecutive run of frames in frame_range where predicate(count)
    is True. Returns {"start_frame", "end_frame", "length"}; length 0 with
    None bounds if no frame satisfies predicate."""
    best_start = best_end = None
    best_len = 0
    cur_start = None
    cur_len = 0
    for f in frame_range:
        count = active_by_frame.get(f, 0)
        if predicate(count):
            if cur_start is None:
                cur_start = f
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = f
        else:
            cur_start = None
            cur_len = 0
    if best_len == 0:
        return {"start_frame": None, "end_frame": None, "length": 0}
    return {"start_frame": best_start, "end_frame": best_end, "length": best_len}


def _median(values):
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def build_temporal_coverage_audit(candidate_tracks, boundary_window=10):
    """AGENTS_MASTER.md section 14, Temporal Candidate Support. Frame range
    is the candidate pool's own observed min..max frame (not an externally
    supplied shot range). boundary_window is the number of frames from each
    end of the range used for opening_boundary_minimum /
    ending_boundary_minimum."""
    active_by_frame = {}
    for t in candidate_tracks:
        for obs in t["observations"]:
            f = obs["production_frame"]
            active_by_frame[f] = active_by_frame.get(f, 0) + 1

    if not active_by_frame:
        return {
            "frame_min": None, "frame_max": None,
            "active_candidate_tracks": {},
            "frames_with_0_active": 0, "frames_with_1_2": 0, "frames_with_3_5": 0,
            "frames_with_6_7": 0, "frames_with_8_plus": 0,
            "min_simultaneous": 0, "median_simultaneous": 0, "max_simultaneous": 0,
            "longest_zero_track_range": {"start_frame": None, "end_frame": None, "length": 0},
            "longest_low_support_range": {"start_frame": None, "end_frame": None, "length": 0},
            "opening_boundary_minimum": 0, "ending_boundary_minimum": 0,
        }

    frame_min = min(active_by_frame)
    frame_max = max(active_by_frame)
    frame_range = list(range(frame_min, frame_max + 1))
    counts = [active_by_frame.get(f, 0) for f in frame_range]

    buckets = {"frames_with_0_active": 0, "frames_with_1_2": 0, "frames_with_3_5": 0,
               "frames_with_6_7": 0, "frames_with_8_plus": 0}
    for c in counts:
        if c == 0:
            buckets["frames_with_0_active"] += 1
        elif c <= 2:
            buckets["frames_with_1_2"] += 1
        elif c <= 5:
            buckets["frames_with_3_5"] += 1
        elif c <= 7:
            buckets["frames_with_6_7"] += 1
        else:
            buckets["frames_with_8_plus"] += 1

    opening_frames = frame_range[:boundary_window]
    ending_frames = frame_range[-boundary_window:]

    return {
        "frame_min": frame_min,
        "frame_max": frame_max,
        "active_candidate_tracks": {str(f): active_by_frame.get(f, 0) for f in frame_range},
        **buckets,
        "min_simultaneous": min(counts),
        "median_simultaneous": _median(counts),
        "max_simultaneous": max(counts),
        "longest_zero_track_range": _longest_run(frame_range, active_by_frame, lambda c: c == 0),
        "longest_low_support_range": _longest_run(frame_range, active_by_frame, lambda c: c <= 2),
        "opening_boundary_minimum": min(active_by_frame.get(f, 0) for f in opening_frames),
        "ending_boundary_minimum": min(active_by_frame.get(f, 0) for f in ending_frames),
    }
