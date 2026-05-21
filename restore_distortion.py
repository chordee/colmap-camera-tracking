import json
import cv2
import numpy as np
import os
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Restore (undistort) images based on camera calibration JSON.")
    parser.add_argument("--undistorted_json", type=str, required=True,
                        help="Path to the undistorted camera JSON (e.g. transforms_undistorted.json). "
                             "Carries the output canvas geometry (K_new) with zeroed distortion coefficients.")
    parser.add_argument("--output_dir", type=str, default="restored_output", help="Directory to save the restored images.")
    parser.add_argument("--image_dir", type=str, default=None, help="Override the directory to search for input images.")
    parser.add_argument("--undistort", action="store_true", help="Undistort images (restore). Default is to apply distortion (reverse).")
    parser.add_argument("--exr", action="store_true", help="Process .exr files in the directory instead of frames in JSON. Default off.")
    parser.add_argument("--original_json", type=str, default=None,
                        help="Path to the original camera JSON (e.g. *_transforms.json) before undistortion. "
                             "Provides K_orig, original canvas size, and real distortion coefficients. "
                             "Required when --undistorted_json has zeroed-out distortion (expand mode).")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load JSON Data
    if not os.path.exists(args.undistorted_json):
        print(f"Error: JSON file not found at {args.undistorted_json}")
        sys.exit(1)

    print(f"Loading calibration data from {args.undistorted_json}...")
    with open(args.undistorted_json, 'r') as f:
        data = json.load(f)

    # 2. Extract Camera Parameters
    try:
        fl_x = float(data['fl_x'])
        fl_y = float(data['fl_y'])
        cx = float(data['cx'])
        cy = float(data['cy'])
        w = int(data['w'])
        h = int(data['h'])
        
        # Distortion coefficients
        k1 = float(data.get('k1', 0))
        k2 = float(data.get('k2', 0))
        k3 = float(data.get('k3', 0))
        k4 = float(data.get('k4', 0))
        p1 = float(data.get('p1', 0))
        p2 = float(data.get('p2', 0))
        
        is_fisheye = data.get('is_fisheye', False)
        
    except KeyError as e:
        print(f"Error: Missing critical key in JSON data: {e}")
        sys.exit(1)

    # 2b. Override distortion coefficients from a separate JSON if provided.
    # Use this when --undistorted_json is transforms_undistorted.json (expand mode):
    # it carries the correct canvas geometry but distortion has been zeroed out.
    # --original_json should point to the original transforms.json which still
    # holds the real k1/k2/k3/k4/p1/p2 values.
    K_orig = None
    w_orig = h_orig = None
    if args.original_json:
        if not os.path.exists(args.original_json):
            print(f"Error: distortion_json not found at {args.original_json}")
            sys.exit(1)
        print(f"Loading distortion coefficients from {args.original_json}...")
        with open(args.original_json, 'r') as f:
            dist_data = json.load(f)
        k1 = float(dist_data.get('k1', 0))
        k2 = float(dist_data.get('k2', 0))
        k3 = float(dist_data.get('k3', 0))
        k4 = float(dist_data.get('k4', 0))
        p1 = float(dist_data.get('p1', 0))
        p2 = float(dist_data.get('p2', 0))
        is_fisheye = dist_data.get('is_fisheye', is_fisheye)
        # Expand-mode reverse: store original canvas geometry for correct re-distortion
        try:
            w_orig     = int(dist_data['w'])
            h_orig     = int(dist_data['h'])
            K_orig = np.array([[float(dist_data['fl_x']), 0,                      float(dist_data['cx'])],
                               [0,                      float(dist_data['fl_y']), float(dist_data['cy'])],
                               [0,                      0,                      1                      ]], dtype=np.float32)
        except KeyError:
            K_orig = None
            w_orig = h_orig = None

    # 3. Prepare File List (Moved up to check resolution)
    # Determine base directory for images
    base_dir = args.image_dir if args.image_dir else os.path.dirname(os.path.abspath(args.undistorted_json))

    # Prepare file list and read flags
    files_to_process = []
    read_flags = cv2.IMREAD_COLOR

    # If --image_dir is provided OR --exr is set, we scan the directory
    if args.image_dir or args.exr:
        if args.exr:
            print(f"EXR Mode Enabled: Scanning {base_dir} for .exr files...")
            read_flags = cv2.IMREAD_UNCHANGED
            valid_exts = ('.exr',)
        else:
            print(f"Scanning {base_dir} for images (ignoring JSON frames list)...")
            valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')

        # Find all files with valid extensions in base_dir
        if os.path.exists(base_dir):
            files_to_process = [f for f in os.listdir(base_dir) if f.lower().endswith(valid_exts)]
            files_to_process.sort() # Ensure consistent order
        else:
            print(f"Error: Image directory {base_dir} does not exist.")
            sys.exit(1)
            
        if not files_to_process:
             print(f"No matching image files found in {base_dir}")
             sys.exit(0)
             
    else:
        # Fallback to JSON frames if no image_dir override
        frames = data.get('frames', [])
        if not frames:
            print("No frames found in JSON 'frames' list.")
            sys.exit(0)
        files_to_process = frames

    # 4. Check Resolution & Scale Intrinsics
    print(f"Checking resolution of first image to verify consistency...")
    
    # Get path of first image
    first_item = files_to_process[0]
    if args.image_dir or args.exr:
        first_img_path = os.path.join(base_dir, first_item)
    else:
        rel_path = first_item['file_path'].replace('\\', os.sep).replace('/', os.sep)
        if rel_path.startswith(f'.{os.sep}'): rel_path = rel_path[2:]
        first_img_path = os.path.join(base_dir, rel_path)

    if os.path.exists(first_img_path):
        # Read header only (or full image if needed) to get size
        temp_img = cv2.imread(first_img_path, read_flags)
        if temp_img is not None:
            real_h, real_w = temp_img.shape[:2]

            # In undistort+original_json mode the input images live in K_orig space,
            # so compare against w_orig/h_orig rather than the K_new canvas.
            if args.undistort and K_orig is not None:
                ref_w, ref_h = w_orig, h_orig
            else:
                ref_w, ref_h = w, h

            if real_w != ref_w or real_h != ref_h:
                print(f"[WARN] Resolution Mismatch Detected!")
                print(f"       Reference Calibration: {ref_w}x{ref_h}")
                print(f"       Actual Image:          {real_w}x{real_h}")

                scale_x = real_w / ref_w
                scale_y = real_h / ref_h

                print(f"       -> Scaling intrinsics by X:{scale_x:.4f}, Y:{scale_y:.4f}")

                # Scale K_new canvas (always)
                fl_x *= scale_x
                fl_y *= scale_y
                cx   *= scale_x
                cy   *= scale_y
                w = int(round(w * scale_x))
                h = int(round(h * scale_y))
                if K_orig is not None:
                    K_orig[0, 0] *= scale_x
                    K_orig[1, 1] *= scale_y
                    K_orig[0, 2] *= scale_x
                    K_orig[1, 2] *= scale_y
                    w_orig = int(round(w_orig * scale_x))
                    h_orig = int(round(h_orig * scale_y))
            else:
                print(f"       Resolution matches ({ref_w}x{ref_h}).")
        else:
            print(f"[WARN] Could not read first image {first_img_path} to verify resolution.")
    else:
        print(f"[WARN] First image not found at {first_img_path}. Proceeding with JSON defaults.")

    # 5. Construct Matrices
    # Camera Matrix (K)
    K = np.array([[fl_x, 0, cx],
                  [0, fl_y, cy],
                  [0, 0, 1]], dtype=np.float32)

    # Distortion Coefficients (D)
    D = np.array([k1, k2, p1, p2, k3, k4, 0, 0], dtype=np.float32)

    # Hardcoded alpha to keep all pixels
    alpha = 1.0

    print(f"  Final Processing Resolution: {w}x{h}")
    print(f"  Camera Matrix (K):\n{K}")
    print(f"  Distortion Coeffs (D):\n{D}")
    print(f"  Model: {'Fisheye' if is_fisheye else 'Perspective'}")
    print(f"  Mode: {'Restore (Undistorting)' if args.undistort else 'Reverse (Distorting)'}")

    # 6. Pre-calculate Maps
    print("Pre-calculating remapping maps...")
    
    if not args.undistort:
        # Reverse Mode (Default): Create Distorted Image from Linear Image
        # We need a map: Dest(Distorted) -> Src(Linear)
        
        # 2. Map Distorted Points -> Linear Points
        if is_fisheye:
            out_w, out_h = w, h
            D_fish = D[:4]
            # Estimate the linear camera matrix used in the undistorted input
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D_fish, (w, h), np.eye(3), balance=alpha
            )
            grid_x, grid_y = np.meshgrid(np.arange(out_w), np.arange(out_h))
            pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float32)
            # undistortPoints: Distorted -> Linear
            pts_u = cv2.fisheye.undistortPoints(pts, K, D_fish, np.eye(3), new_K)
        else:
            # Standard Perspective
            if K_orig is not None:
                # Expand-mode: output in K_orig space (original size), sample from K_new space
                out_w, out_h = w_orig, h_orig
                K_dist   = K_orig
                K_undist = K        # K_new from transforms_undistorted.json
                print(f"  [Expand-mode reverse] Output: {out_w}x{out_h} (original size)")
            else:
                out_w, out_h = w, h
                K_dist   = K
                K_undist = K.copy()
            grid_x, grid_y = np.meshgrid(np.arange(out_w), np.arange(out_h))
            pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2).astype(np.float32)
            pts_u = cv2.undistortPoints(pts, K_dist, D, None, K_undist)

        map_coords = pts_u.reshape(out_h, out_w, 2)
        map1, map2 = cv2.convertMaps(map_coords[..., 0], map_coords[..., 1], cv2.CV_16SC2, nninterpolation=False)
        
    else:
        # Normal Mode (Undistort): Create Linear Image from Distorted Image
        # We need a map: Dest(Linear) -> Src(Distorted)
        
        if is_fisheye:
            D_fish = D[:4]
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D_fish, (w, h), np.eye(3), balance=alpha
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D_fish, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
        else:
            if K_orig is not None:
                # Expand-mode: input in K_orig space (distorted), output in K_new space
                map1, map2 = cv2.initUndistortRectifyMap(
                    K_orig, D, None, K, (w, h), cv2.CV_16SC2
                )
                print(f"  [Expand-mode undistort] Output: {w}x{h}")
            else:
                new_K = K.copy()
                map1, map2 = cv2.initUndistortRectifyMap(
                    K, D, None, new_K, (w, h), cv2.CV_16SC2
                )

    # The shared remap maps assume a specific input size. In undistort + K_orig
    # (expand-mode forward) the source images live in the K_orig canvas; every
    # other path samples from the K_new canvas.
    if args.undistort and K_orig is not None:
        expected_in_w, expected_in_h = w_orig, h_orig
    else:
        expected_in_w, expected_in_h = w, h

    # 7. Prepare Output Directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # 8. Process Images
    print(f"Starting processing of {len(files_to_process)} images...")

    for i, item in enumerate(files_to_process):
        # Determine image path based on mode
        if args.image_dir or args.exr:
            # item is just the filename
            rel_path = item
            image_path = os.path.join(base_dir, rel_path)
        else:
            # item is a frame dict from JSON
            rel_path = item['file_path']
            rel_path = rel_path.replace('\\', os.sep).replace('/', os.sep)
            
            # If path starts with ./, remove it to join cleanly
            if rel_path.startswith(f'.{os.sep}'):
                rel_path = rel_path[2:]
                
            image_path = os.path.join(base_dir, rel_path)
        
        if not os.path.exists(image_path):
            print(f"  [Skipping] Image not found: {image_path}")
            continue

        # Read image
        img = cv2.imread(image_path, read_flags)
        if img is None:
            print(f"  [Skipping] Could not read image: {image_path}")
            continue

        # The shared remap maps only work for images matching the expected input
        # size. A mismatched frame would sample garbage, so skip it with a clear
        # warning rather than emit corrupted output.
        if img.shape[1] != expected_in_w or img.shape[0] != expected_in_h:
            print(f"  [Skipping] {os.path.basename(image_path)}: size "
                  f"{img.shape[1]}x{img.shape[0]} differs from expected "
                  f"{expected_in_w}x{expected_in_h}; cannot reuse shared maps.")
            continue

        # Perform Remapping (works for both directions now)
        processed_img = cv2.remap(
            img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        # Save result
        filename = os.path.basename(image_path)
        save_path = os.path.join(args.output_dir, filename)
        
        save_params = []
        if filename.lower().endswith('.exr'):
            # Attempt to match the output EXR type to the processing data type
            if processed_img.dtype == np.float32:
                save_params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
            elif processed_img.dtype == np.float16:
                save_params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]

        cv2.imwrite(save_path, processed_img, save_params)
        
        # Simple progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(files_to_process)} images...")

    print("Processing complete.")

if __name__ == "__main__":
    main()
