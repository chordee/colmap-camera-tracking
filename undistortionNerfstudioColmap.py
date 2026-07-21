import json
import cv2
import numpy as np
import os
import sys
import math
from pathlib import Path
import argparse


KEEP_DISTORTION_ALLOWED_MODELS = {"OPENCV", "FULL_OPENCV"}


def validate_keep_distortion_model(camera_model):
    """Raise ValueError if camera_model's distortion formula doesn't match
    Houdini's kma_physicallens OpenCV section (Brown-Conrady radial/tangential,
    not the angle-based fisheye/equidistant formula)."""
    if camera_model not in KEEP_DISTORTION_ALLOWED_MODELS:
        raise ValueError(
            f"--keep-distortion requires camera_model to be one of "
            f"{sorted(KEEP_DISTORTION_ALLOWED_MODELS)}; got {camera_model!r}. "
            "Fisheye models use an angle-based distortion formula that does not "
            "match Houdini's kma_physicallens OpenCV section, which implements "
            "the Brown-Conrady radial/tangential model."
        )


def build_keep_distortion_json(data, json_dir, output_dir):
    """Build the pass-through JSON for --keep-distortion mode: same intrinsics
    and distortion coefficients as the source JSON, with frames[].file_path
    rewritten to be relative to output_dir instead of json_dir. No image data
    is read or written -- the original frames are referenced in place."""
    new_data = data.copy()
    new_frames = []
    for frame in data.get("frames", []):
        rel_path = frame["file_path"]
        abs_path = (Path(json_dir) / rel_path).resolve()
        new_rel_path = os.path.relpath(abs_path, output_dir).replace(os.sep, '/')
        new_frame = frame.copy()
        new_frame["file_path"] = new_rel_path
        new_frames.append(new_frame)
    new_data["frames"] = new_frames
    return new_data


def keep_distortion_process(json_path, output_dir):
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)

    print(f"Reading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    camera_model = data.get("camera_model")
    try:
        validate_keep_distortion_model(camera_model)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_dir = Path(json_path).parent

    new_data = build_keep_distortion_json(data, json_dir, output_path)

    new_json_path = output_path / "transforms_original.json"
    with open(new_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)

    print("Done! Distortion coefficients preserved -- images were not rectified.")
    print(f"Original images referenced from : {json_dir}")
    print(f"New JSON saved to                : {new_json_path}")
    print("Use this new JSON in Houdini -- the camera node's distortion spare "
          "parameters can be copied into a kma_physicallens VOP.")


def compute_undistorted_canvas(w, h, K, D, n_samples=50):
    """
    Compute the output canvas size needed to contain all border pixels after
    undistortion, without changing the focal length.

    Sample points along all four edges of the distorted image, compute their
    undistorted positions via undistortPoints, and take the bounding box.
    The resulting canvas is guaranteed to include the original [0, w-1] x [0, h-1]
    range as well.

    Returns: (x_min, y_min, new_w, new_h, new_cx, new_cy)
      - x_min, y_min : top-left corner of the bounding box in undistorted
                       pixel coordinates (may be negative for barrel distortion)
      - new_w, new_h : expanded canvas dimensions in pixels
      - new_cx, new_cy : principal point position in the new canvas
    """
    pts = []
    for t in np.linspace(0, 1, n_samples):
        pts.extend([
            [t * (w - 1), 0          ],
            [t * (w - 1), h - 1      ],
            [0,           t * (h - 1)],
            [w - 1,       t * (h - 1)],
        ])
    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    # P=K outputs back into pixel coordinates
    undist_pts = cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)

    # Bounding box, ensuring the original canvas range is always included
    x_min = min(0.0, float(undist_pts[:, 0].min()))
    y_min = min(0.0, float(undist_pts[:, 1].min()))
    x_max = max(float(w - 1), float(undist_pts[:, 0].max()))
    y_max = max(float(h - 1), float(undist_pts[:, 1].max()))

    new_w = int(np.ceil(x_max - x_min)) + 1
    new_h = int(np.ceil(y_max - y_min)) + 1

    # Focal length unchanged; principal point shifts with the canvas origin
    new_cx = float(K[0, 2]) - x_min
    new_cy = float(K[1, 2]) - y_min

    return x_min, y_min, new_w, new_h, new_cx, new_cy


def undistort_process(json_path, output_dir, crop=False):
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return

    output_path = Path(output_dir)
    images_out_dir = output_path / "images_undistorted"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    camera_model = data.get("camera_model", "OPENCV")
    if camera_model == "OPENCV_FISHEYE":
        print("Error: OPENCV_FISHEYE camera model is not supported. "
              "Use cv2.fisheye functions for equidistant distortion.")
        return

    # 1. Camera intrinsics
    w    = int(data.get("w", 1920))
    h    = int(data.get("h", 1080))
    fl_x = float(data.get("fl_x", 1000))
    fl_y = float(data.get("fl_y", fl_x))
    cx   = float(data.get("cx", w / 2))
    cy   = float(data.get("cy", h / 2))

    # 2. Distortion coefficients
    k1 = float(data.get("k1", 0.0))
    k2 = float(data.get("k2", 0.0))
    k3 = float(data.get("k3", 0.0))
    k4 = float(data.get("k4", 0.0))
    p1 = float(data.get("p1", 0.0))
    p2 = float(data.get("p2", 0.0))

    K = np.array([
        [fl_x, 0,    cx],
        [0,    fl_y, cy],
        [0,    0,    1 ]
    ], dtype=np.float64)

    # OpenCV distortion vector order: k1 k2 p1 p2 k3 k4
    D = np.array([k1, k2, p1, p2, k3, k4, 0.0, 0.0], dtype=np.float64)

    print(f"Camera Matrix:\n{K}")
    print(f"Distortion Coeffs: {D}")

    # 3. Determine output canvas
    if crop:
        # Keep the original canvas size (w x h).
        # Focal length, cx, and cy are unchanged.  Corner regions that fall
        # outside the distorted image will appear black, but focal length and
        # aperture in Houdini stay at their nominal physical values.
        new_w, new_h   = w, h
        new_cx, new_cy = cx, cy
        print(f"Mode: crop  -- canvas unchanged: {new_w} x {new_h}")
    else:
        # Expand the canvas so that every pixel from the distorted image is
        # visible after undistortion.  Focal length is preserved; cx/cy shift
        # to match the new canvas origin.
        _, _, new_w, new_h, new_cx, new_cy = compute_undistorted_canvas(w, h, K, D)
        print(f"Mode: expand -- canvas: {w} x {h}  ->  {new_w} x {new_h},  "
              f"cx: {cx:.2f} -> {new_cx:.2f},  cy: {cy:.2f} -> {new_cy:.2f}")

    new_K = np.array([
        [fl_x, 0,    new_cx],
        [0,    fl_y, new_cy],
        [0,    0,    1     ]
    ], dtype=np.float64)

    # Build remap tables once; all frames share the same intrinsics
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (new_w, new_h), cv2.CV_32FC1)

    # 4. Update JSON camera parameters
    new_data = data.copy()
    new_data["w"]  = new_w
    new_data["h"]  = new_h
    new_data["cx"] = new_cx
    new_data["cy"] = new_cy
    # Original sensor dimensions, kept for physical focal-length recovery in Houdini
    new_data["sensor_w"] = w
    new_data["sensor_h"] = h
    # Updated field of view (symmetric approximation; Houdini uses fl_x/cx/w directly)
    new_data["camera_angle_x"] = 2 * math.atan(new_w / (2 * fl_x))
    new_data["camera_angle_y"] = 2 * math.atan(new_h / (2 * fl_y))
    # Zero out distortion coefficients (images are now undistorted)
    for key in ["k1", "k2", "k3", "k4", "p1", "p2"]:
        new_data[key] = 0.0

    # 5. Process frames
    json_dir = Path(json_path).parent
    frames = data.get("frames", [])
    print(f"Processing {len(frames)} images...")

    new_frames = []
    for idx, frame in enumerate(frames):
        rel_path = frame["file_path"]
        img_path = json_dir / rel_path

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Failed to read image: {img_path}")
            continue

        # Remap to the expanded canvas; black fill outside the original image area
        dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        img_name = Path(rel_path).name
        save_path = images_out_dir / img_name
        if save_path.exists():
            print(f"Warning: Duplicate filename, overwriting: {img_name}")
        cv2.imwrite(str(save_path), dst)

        new_frame = frame.copy()
        new_frame["file_path"] = f"images_undistorted/{img_name}"
        new_frames.append(new_frame)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(frames):
            print(f"Processed {idx + 1}/{len(frames)}...")

    new_data["frames"] = new_frames

    # 6. Save updated JSON
    new_json_path = output_path / "transforms_undistorted.json"
    with open(new_json_path, 'w') as f:
        json.dump(new_data, f, indent=4)

    print("Done!")
    print(f"Undistorted images saved to : {images_out_dir}")
    print(f"New JSON saved to           : {new_json_path}")
    print("Use this new JSON in Houdini!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Undistort images and transforms.json")
    parser.add_argument("--original_json", type=str, required=True,
                        help="Path to the original transforms.json (pre-undistortion). "
                             "Same naming as restore_distortion.py's --original_json.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--crop", action="store_true",
                        help="Keep original canvas size instead of expanding it.  "
                             "Focal length and aperture in Houdini remain at their "
                             "nominal physical values (e.g. 20 mm / 36 mm).  "
                             "Corner pixels that fall outside the distorted image "
                             "will appear black.")
    parser.add_argument("--keep-distortion", dest="keep_distortion", action="store_true",
                        help="Skip rectification entirely and pass the original "
                             "distorted images plus OpenCV distortion coefficients "
                             "through to Houdini instead. Only valid for camera_model "
                             "OPENCV or FULL_OPENCV.")
    args = parser.parse_args()
    if args.keep_distortion:
        keep_distortion_process(args.original_json, args.output_dir)
    else:
        undistort_process(args.original_json, args.output_dir, crop=args.crop)
