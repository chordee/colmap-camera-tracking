import math
import os
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
