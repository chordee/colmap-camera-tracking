import os
import random
import re


class SceneLoadError(Exception):
    """Raised when a scene folder is missing required COLMAP text files."""


def _get_frame_num(file_path):
    """Extract the first run of digits from a filename -- e.g.
    'frame_000001.jpg' -> 1."""
    fname = os.path.basename(file_path)
    match = re.search(r'(\d+)', fname)
    return int(match.group(1)) if match else 0


def load_image_size(cameras_path):
    """Read (width, height) from COLMAP's sparse/0/cameras.txt -- the
    original, undistorted-pipeline-untouched image size. Rejects scenes
    with cameras of differing dimensions: write_3de_2d_tracks_txt's Y-flip
    uses one global height, so a mixed-dimension scene would silently
    apply the wrong height to some observations' Y coordinates."""
    sizes = []
    with open(cameras_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            sizes.append((int(float(parts[2])), int(float(parts[3]))))
    if not sizes:
        raise SceneLoadError(f"No camera found in {cameras_path}")
    if len(set(sizes)) > 1:
        raise SceneLoadError(
            f"Multiple cameras with different image dimensions found in "
            f"{cameras_path} ({sorted(set(sizes))}) -- mixed-dimension "
            f"scenes are not supported."
        )
    return sizes[0]


def build_tracks(images_path):
    """Read sparse/0/images.txt and build persistent 2D tracks keyed by
    COLMAP's POINT3D_ID. Every track with a valid persistent identity
    (POINT3D_ID != -1) is retained, including ones with a same-track/
    same-frame conflict -- conflicting tracks get structural_status
    "CONFLICT" and keep every raw, conflicting observation (never
    averaged, never a single "winner" picked). Returns a list of track
    dicts, sorted by track_id, each observations list sorted by
    production_frame (a CONFLICT track may have multiple entries sharing
    the same production_frame -- intentional). duplicate_status/
    duplicate_of start at "UNIQUE"/None; classify_exact_duplicates refines
    them. structural_status starts at "CONFLICT"/"VALID" only;
    classify_structural_status refines "VALID" further."""
    observations_by_point = {}
    seen_frames_by_point = {}
    conflict_ids = set()

    with open(images_path, "r") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]

    i = 0
    while i < len(lines):
        header = lines[i].split()
        name = header[9]
        frame_num = _get_frame_num(name)
        i += 1
        points_line = lines[i] if i < len(lines) else ""
        i += 1

        parts = points_line.split()
        for j in range(0, len(parts), 3):
            x = float(parts[j])
            y = float(parts[j + 1])
            point3d_id = parts[j + 2]
            if point3d_id == "-1":
                continue
            seen = seen_frames_by_point.setdefault(point3d_id, set())
            if frame_num in seen:
                conflict_ids.add(point3d_id)
            else:
                seen.add(frame_num)
            observations_by_point.setdefault(point3d_id, []).append((frame_num, x, y, name))

    tracks = []
    for point3d_id, obs in observations_by_point.items():
        obs.sort(key=lambda o: o[0])
        tracks.append({
            "track_id": f"colmap::{point3d_id}",
            "track_name": f"p{point3d_id}",
            "observations": [
                {"production_frame": f, "x": x, "y": y, "image_name": name}
                for f, x, y, name in obs
            ],
            "structural_status": "CONFLICT" if point3d_id in conflict_ids else "VALID",
            "duplicate_status": "UNIQUE",
            "duplicate_of": None,
        })
    tracks.sort(key=lambda t: t["track_id"])
    return tracks


def classify_structural_status(tracks, scene_dir, width, height):
    """Refine each non-CONFLICT track's structural_status by checking, over
    all of its observations: coordinates inside the image domain, the
    referenced image file exists on disk, and the frame number was
    successfully parsed. Precedence when a track fails more than one check:
    INVALID_IMAGE_MISSING > INVALID_COORDS > INVALID_FRAME (CONFLICT tracks,
    already tagged by build_tracks, are left untouched -- highest
    precedence). Returns a new list; does not mutate the input.

    If scene_dir's images/ subdirectory does not exist at all, the
    image-existence check is skipped for every track (they are not
    penalized for it). This is intentional, not an oversight: real COLMAP
    scene folders always ship an images/ directory, so the skip only
    protects against a malformed/incomplete scene folder rather than
    silently failing the whole classification pass.

    image_name may be a relative path with subdirectories (COLMAP records
    images.txt NAME entries as paths relative to the images/ root when the
    source images were organized in subfolders), so the on-disk listing is
    built by walking images_dir recursively rather than a flat listdir."""
    images_dir = os.path.join(scene_dir, "images")
    if os.path.isdir(images_dir):
        existing_images = set()
        for root, _dirs, files in os.walk(images_dir):
            rel_root = os.path.relpath(root, images_dir)
            for fname in files:
                rel_path = fname if rel_root == "." else os.path.join(rel_root, fname)
                existing_images.add(rel_path.replace(os.sep, "/"))
    else:
        existing_images = None

    result = []
    for t in tracks:
        new_t = dict(t)
        if new_t["structural_status"] == "CONFLICT":
            result.append(new_t)
            continue

        # Only check image existence if we have an images directory
        image_missing = False
        if existing_images is not None:
            image_missing = any(
                obs["image_name"].replace(os.sep, "/") not in existing_images
                for obs in t["observations"]
            )

        coords_bad = any(
            not (0 <= obs["x"] < width and 0 <= obs["y"] < height)
            for obs in t["observations"]
        )
        frame_bad = any(obs["production_frame"] == 0 for obs in t["observations"])

        if image_missing:
            new_t["structural_status"] = "INVALID_IMAGE_MISSING"
        elif coords_bad:
            new_t["structural_status"] = "INVALID_COORDS"
        elif frame_bad:
            new_t["structural_status"] = "INVALID_FRAME"
        else:
            new_t["structural_status"] = "VALID"
        result.append(new_t)
    return result


def classify_exact_duplicates(tracks):
    """Detect exact-duplicate tracks (identical canonical observation
    sequences: same count, same (production_frame, x, y) values, no
    rounding, no tolerance) across ALL tracks regardless of
    structural_status. Grouping by the canonical tuple as a dict key is
    O(n) (Python's dict hashing/equality handles this directly -- no manual
    bucketing needed). Within each duplicate group, PRIMARY = smallest
    point3d_id (compared numerically); every other member gets
    duplicate_status="EXACT_DUPLICATE" and duplicate_of=<PRIMARY track_id>.
    Returns a new list; does not mutate the input."""
    def canonical_key(t):
        return tuple(sorted(
            (o["production_frame"], o["x"], o["y"]) for o in t["observations"]
        ))

    def point_id_num(track_id):
        return int(track_id.split("::", 1)[1])

    groups = {}
    for t in tracks:
        groups.setdefault(canonical_key(t), []).append(t["track_id"])

    result = [dict(t) for t in tracks]
    by_track_id = {t["track_id"]: t for t in result}

    for track_ids in groups.values():
        if len(track_ids) < 2:
            continue
        primary_id = min(track_ids, key=point_id_num)
        for tid in track_ids:
            if tid != primary_id:
                by_track_id[tid]["duplicate_status"] = "EXACT_DUPLICATE"
                by_track_id[tid]["duplicate_of"] = primary_id

    return result


def is_exportable(track):
    """A track is eligible for 3DE export only if it's structurally clean
    and not a duplicate secondary. (This predicate anticipates Issue 03's
    formal "Production Candidate Pool" definition -- Persistent Track AND
    structurally_valid AND non-conflict AND not EXACT_DUPLICATE -- applied
    now since there's no reason to export known-bad or redundant data.)"""
    return track["structural_status"] == "VALID" and track["duplicate_status"] == "UNIQUE"


def write_structural_qc_report(tracks, out_path):
    """Write a human-readable structural QC report: summary counts per
    structural_status, an exact-duplicate count, then a listing of only
    the non-clean tracks (not VALID, or an EXACT_DUPLICATE secondary) --
    a full listing of every track would be unreadable at this pipeline's
    real scale (100k+ tracks on a typical scene)."""
    status_counts = {}
    for t in tracks:
        status_counts[t["structural_status"]] = status_counts.get(t["structural_status"], 0) + 1
    duplicate_count = sum(1 for t in tracks if t["duplicate_status"] == "EXACT_DUPLICATE")

    lines = [f"Total tracks: {len(tracks)}"]
    for status in sorted(status_counts):
        lines.append(f"  {status}: {status_counts[status]}")
    lines.append(f"Exact duplicate tracks (secondaries): {duplicate_count}")
    lines.append("")
    lines.append("Non-clean tracks:")
    non_clean = [
        t for t in tracks
        if t["structural_status"] != "VALID" or t["duplicate_status"] == "EXACT_DUPLICATE"
    ]
    if not non_clean:
        lines.append("  (none)")
    else:
        for t in non_clean:
            lines.append(
                f"  {t['track_id']}: structural_status={t['structural_status']}, "
                f"duplicate_status={t['duplicate_status']}, duplicate_of={t['duplicate_of']}"
            )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_scene(scene_dir):
    """Read a processed COLMAP tracking scene folder's raw (undistorted-
    pipeline-untouched) 2D feature observations, build persistent tracks,
    and classify each one's structural_status and duplicate_status."""
    cameras_path = os.path.join(scene_dir, "sparse", "0", "cameras.txt")
    images_path = os.path.join(scene_dir, "sparse", "0", "images.txt")

    if not os.path.exists(cameras_path):
        raise SceneLoadError(f"Not found: {cameras_path}")
    if not os.path.exists(images_path):
        raise SceneLoadError(f"Not found: {images_path}")

    width, height = load_image_size(cameras_path)
    tracks = build_tracks(images_path)
    tracks = classify_structural_status(tracks, scene_dir, width, height)
    tracks = classify_exact_duplicates(tracks)

    return {
        "width": width, "height": height,
        "tracks": tracks,
    }


def filter_by_min_observations(tracks, min_observations):
    """Drop tracks with fewer than min_observations total observations -- a
    track's observation count is its 'existence duration' proxy; shorter
    tracks are more likely to be spurious COLMAP feature matches.
    None/0 disables filtering."""
    if not min_observations:
        return tracks
    return [t for t in tracks if len(t["observations"]) >= min_observations]


def sample_tracks(tracks, max_tracks, seed=None):
    """Return tracks unchanged when max_tracks is None/0 (no limit);
    otherwise a uniform random sample of at most max_tracks, without
    replacement."""
    if not max_tracks or max_tracks >= len(tracks):
        return tracks
    rng = random.Random(seed)
    return rng.sample(tracks, max_tracks)


def write_3de_2d_tracks_txt(tracks, production_start_frame, image_height, out_path):
    """Write 3DEqualizer's native 2D Tracks ASCII format (ADAPTER_3DE_R5.md
    v1 contract, cross-verified against the real bundled export_tracks.py):
    TRACK_COUNT, then per track: name / static field "0" (verified-working
    native format field; semantic meaning not confirmed upstream, must not
    be promoted to a semantic concept here either) / sample count / one
    "<frame> <x> <y>" line per observation. Zero blank lines, zero comments,
    no indentation -- 3DE's grammar is exact-whitespace sensitive.

    COLMAP's coordinates are top-left-origin, Y-down. This 3DE format's Y is
    bottom-left-origin, Y-up. Y is flipped as y_3de = image_height - y_colmap
    for every observation to match 3DE's native convention."""
    lines = [str(len(tracks))]
    for t in tracks:
        lines.append(t["track_name"])
        lines.append("0")
        lines.append(str(len(t["observations"])))
        for obs in t["observations"]:
            frame_3de = obs["production_frame"] - production_start_frame + 1
            if frame_3de < 1:
                raise ValueError(
                    f"production_frame {obs['production_frame']} is before "
                    f"production_start_frame {production_start_frame} "
                    f"(resulting 3DE frame {frame_3de} < 1)"
                )
            y_3de = image_height - obs["y"]
            lines.append(f"{frame_3de} {obs['x']:.15f} {y_3de:.15f}")

    with open(out_path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
