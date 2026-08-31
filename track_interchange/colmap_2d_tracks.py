import math
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
    original, undistorted-pipeline-untouched image size (assumes a single
    shared camera, this pipeline's default single_camera COLMAP setting)."""
    with open(cameras_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            return int(float(parts[2])), int(float(parts[3]))
    raise SceneLoadError(f"No camera found in {cameras_path}")


def build_tracks(images_path):
    """Read sparse/0/images.txt and build persistent 2D tracks keyed by
    COLMAP's POINT3D_ID. Returns (tracks, conflict_count):
    - tracks: list of {"track_id","track_name","observations"} dicts,
      sorted by track_id, each observations list sorted by production_frame.
    - conflict_count: number of point3d_ids dropped because the same
      (point3d_id, production_frame) pair appeared more than once.
    POINT3D_ID == -1 (untriangulated 2D feature) is excluded."""
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
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            seen = seen_frames_by_point.setdefault(point3d_id, set())
            if frame_num in seen:
                conflict_ids.add(point3d_id)
                continue
            seen.add(frame_num)
            observations_by_point.setdefault(point3d_id, []).append((frame_num, x, y))

    tracks = []
    for point3d_id, obs in observations_by_point.items():
        if point3d_id in conflict_ids:
            continue
        obs.sort(key=lambda o: o[0])
        tracks.append({
            "track_id": f"colmap::{point3d_id}",
            "track_name": f"p{point3d_id}",
            "observations": [
                {"production_frame": f, "x": x, "y": y} for f, x, y in obs
            ],
        })
    tracks.sort(key=lambda t: t["track_id"])
    return tracks, len(conflict_ids)


def load_scene(scene_dir):
    """Read a processed COLMAP tracking scene folder's raw (undistorted-
    pipeline-untouched) 2D feature observations and build persistent tracks."""
    cameras_path = os.path.join(scene_dir, "sparse", "0", "cameras.txt")
    images_path = os.path.join(scene_dir, "sparse", "0", "images.txt")

    if not os.path.exists(cameras_path):
        raise SceneLoadError(f"Not found: {cameras_path}")
    if not os.path.exists(images_path):
        raise SceneLoadError(f"Not found: {images_path}")

    width, height = load_image_size(cameras_path)
    tracks, conflict_count = build_tracks(images_path)

    return {
        "width": width, "height": height,
        "tracks": tracks, "conflict_count": conflict_count,
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
