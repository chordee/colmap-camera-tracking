import json
import os
import re


class SceneLoadError(Exception):
    """Raised when a scene folder is missing required files or has an
    unsupported shape (e.g. multi-camera)."""


def _get_frame_num(file_path):
    fname = os.path.basename(file_path)
    match = re.search(r'(\d+)', fname)
    return int(match.group(1)) if match else 0


def _load_frames(data):
    frames = []
    for frame_data in data.get("frames", []):
        frames.append({
            "frame_num": _get_frame_num(frame_data["file_path"]),
            "transform_matrix": frame_data["transform_matrix"],
        })
    frames.sort(key=lambda f: f["frame_num"])
    return frames


def _load_points(points3d_path):
    points = []
    with open(points3d_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            points.append({
                "id": parts[0],
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
            })
    return points


def load_scene(scene_dir):
    """Read a processed COLMAP tracking scene folder (the output of
    run_autotracker.py's default, rectified pipeline) and return its camera
    intrinsics, per-frame transforms, and sparse point cloud."""
    json_path = os.path.join(scene_dir, "undistort", "transforms_undistorted.json")
    points3d_path = os.path.join(scene_dir, "sparse", "0", "points3D.txt")

    if not os.path.exists(json_path):
        raise SceneLoadError(
            f"Not found: {json_path}\n"
            "This scene must be processed by the default (rectified) pipeline "
            "-- --keep-distortion scenes are not supported."
        )
    if not os.path.exists(points3d_path):
        raise SceneLoadError(f"Not found: {points3d_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    required_keys = ("fl_x", "fl_y", "cx", "cy", "w", "h")
    if not all(k in data for k in required_keys):
        raise SceneLoadError(
            "This JSON does not have shared top-level camera intrinsics "
            "(fl_x/fl_y/cx/cy/w/h) -- multi-camera scenes are not supported."
        )

    return {
        "w": float(data["w"]), "h": float(data["h"]),
        "sensor_w": float(data.get("sensor_w", data["w"])),
        "sensor_h": float(data.get("sensor_h", data["h"])),
        "fl_x": float(data["fl_x"]), "fl_y": float(data["fl_y"]),
        "cx": float(data["cx"]), "cy": float(data["cy"]),
        "frames": _load_frames(data),
        "points": _load_points(points3d_path),
    }
