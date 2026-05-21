import argparse
import os
import subprocess
import sys
import shutil

def main():
    parser = argparse.ArgumentParser(description="Batch runner for autotracker and colmap conversion.")
    parser.add_argument("input_path", help="Path to input directory (videos)")
    parser.add_argument("output_path", help="Path to output directory")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale argument (default: 0.5)")
    parser.add_argument("--overlap", type=int, default=12, help="Sequential matching overlap (default: 12)")
    parser.add_argument("--skip-houdini", action="store_true", help="Skip Houdini scene generation")
    parser.add_argument("--hfs", help="Path to Houdini installation (optional)")
    parser.add_argument("--multi-cams", action="store_true", help="Allow processing multiple videos with different camera settings")
    parser.add_argument("--acescg", action="store_true", help="Convert input ACEScg colorspace to sRGB")
    parser.add_argument("--lut", help="Path to .cube LUT file for color conversion (optional)")
    parser.add_argument("--mask", help="Path to mask directory root (optional)")
    parser.add_argument("--camera_model", help="Specify COLMAP camera model (e.g., OPENCV, PINHOLE, SIMPLE_RADIAL). Default: Auto (COLMAP decides)")
    parser.add_argument("--loop", action="store_true", help="Enable COLMAP loop detection in sequential matching")
    parser.add_argument("--loop_period", type=int, default=5, help="COLMAP loop detection period (default: 5)")
    parser.add_argument("--loop_num_images", type=int, default=50, help="COLMAP loop detection number of images (default: 50)")
    parser.add_argument("--vocab_tree_path", default="vocab_tree_faiss_flickr100K_words32K.bin", help="Path to vocabulary tree for loop detection")
    parser.add_argument("--extra_fe", help="Extra arguments for feature extraction (JSON string or path to .json file)")
    parser.add_argument("--extra_sm", help="Extra arguments for sequential matching (JSON string or path to .json file)")
    parser.add_argument("--extra_ma", help="Extra arguments for mapping (JSON string or path to .json file)")
    parser.add_argument("--focal_length_mm", type=float, default=None, help="Lens focal length in mm (e.g. 24). Combined with --sensor_width_mm to set COLMAP camera_params.")
    parser.add_argument("--sensor_width_mm", type=float, default=36.0, help="Sensor width in mm (default: 36.0 full-frame). Common values: ARRI LF=36.7, Super35=24.89, MFT=17.3")
    parser.add_argument("--crop", action="store_true", help="Keep original canvas size during undistortion instead of expanding it. Focal length and aperture in Houdini remain at their nominal physical values.")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    scale = args.scale

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path, exist_ok=True)
            print(f"[INFO] Created output directory: {output_path}")
        except OSError as e:
            print(f"[ERROR] Could not create output directory: {e}")
            sys.exit(1)

    # Locate autotracker.py (assumed to be in the same directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autotracker_script = os.path.join(script_dir, "autotracker.py")

    # Command 1: python autotracker.py <input_path> <output_path> --scale <scale>
    cmd1 = [sys.executable, autotracker_script, input_path, output_path, "--scale", str(scale), "--overlap", str(args.overlap)]
    if args.multi_cams:
        cmd1.append("--multi-cams")
    if args.acescg:
        cmd1.append("--acescg")
    if args.lut:
        cmd1.extend(["--lut", args.lut])
    if args.mask:
        cmd1.extend(["--mask", args.mask])
    if args.camera_model:
        cmd1.extend(["--camera_model", args.camera_model])
    if args.loop:
        cmd1.append("--loop")
        cmd1.extend(["--loop_period", str(args.loop_period)])
        cmd1.extend(["--loop_num_images", str(args.loop_num_images)])
        if args.vocab_tree_path:
            cmd1.extend(["--vocab_tree_path", args.vocab_tree_path])
            
    if args.extra_fe:
        cmd1.extend(["--extra_fe", args.extra_fe])
    if args.extra_sm:
        cmd1.extend(["--extra_sm", args.extra_sm])
    if args.extra_ma:
        cmd1.extend(["--extra_ma", args.extra_ma])
    if args.focal_length_mm:
        cmd1.extend(["--focal_length_mm", str(args.focal_length_mm)])
        cmd1.extend(["--sensor_width_mm", str(args.sensor_width_mm)])
        
    print(f"Running: {' '.join(cmd1)}")
    try:
        subprocess.run(cmd1, check=True)
    except subprocess.CalledProcessError:
        print("[ERROR] autotracker.py failed.")
        sys.exit(1)

    # A scene only counts as ready for downstream steps when sparse/0/cameras.bin
    # exists — matching the completion marker used by autotracker.py.
    def _has_reconstruction(folder_path):
        return os.path.exists(os.path.join(folder_path, "sparse", "0", "cameras.bin"))

    # Command 2: Run colmap model_converter on subfolders
    print("Scanning output directory for subfolders to convert models...")
    subfolders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]

    for folder in subfolders:
        folder_path = os.path.join(output_path, folder)
        if not _has_reconstruction(folder_path):
            print(f"Skipping {folder}: no completed COLMAP reconstruction.")
            continue
        sparse_0_path = os.path.join(folder_path, "sparse", "0")
        ply_output_path = os.path.join(folder_path, "points3D.ply")

        cmd2 = ["colmap", "model_converter", "--input_path", sparse_0_path, "--output_path", ply_output_path, "--output_type", "PLY"]
        print(f"Running: {' '.join(cmd2)}")
        try:
            subprocess.run(cmd2, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] colmap model_converter failed for {folder}.")

    # Command 3: Copy colmap2nerf.py to output_path
    colmap2nerf_src = os.path.join(script_dir, "colmap2nerf.py")
    colmap2nerf_dst = os.path.join(output_path, "colmap2nerf.py")
    print(f"Copying {colmap2nerf_src} to {colmap2nerf_dst}")
    try:
        shutil.copy(colmap2nerf_src, colmap2nerf_dst)
    except OSError as e:
        print(f"[ERROR] Failed to copy colmap2nerf.py: {e}")
        sys.exit(1)

    # Command 4: Switch workspace and run colmap2nerf on subfolders
    original_cwd = os.getcwd()
    os.chdir(output_path)
    print(f"Switched workspace to: {os.getcwd()}")

    generated_jsons = []
    subfolders = [f for f in os.listdir(".") if os.path.isdir(f)]
    for folder in subfolders:
        if not _has_reconstruction(folder):
            print(f"Skipping {folder}: no completed COLMAP reconstruction.")
            continue
        print(f"Processing folder: {folder}")
        json_filename = f"{folder}_transforms.json"
        cmd_nerf = [
            sys.executable, "colmap2nerf.py",
            "--colmap_db", os.path.join(folder, "database.db"),
            "--images", os.path.join(folder, "images"),
            "--text", os.path.join(folder, "sparse"),
            "--out", json_filename,
            "--keep_colmap_coords"
        ]
        print(f"Running: {' '.join(cmd_nerf)}")
        try:
            subprocess.run(cmd_nerf, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] colmap2nerf.py failed for {folder}; downstream steps will skip it.")
            continue

        if os.path.exists(json_filename):
            generated_jsons.append((os.path.abspath(json_filename), folder))

    # Command 5: Back to original workspace and run undistortion
    os.chdir(original_cwd)
    print(f"Switched workspace back to: {os.getcwd()}")

    undistortion_script = os.path.join(script_dir, "undistortionNerfstudioColmap.py")

    for json_path, folder_name in generated_jsons:
        undistort_output_dir = os.path.join(output_path, folder_name, "undistort")
        cmd_undistort = [
            sys.executable, undistortion_script,
            "--json_path", json_path,
            "--output_dir", undistort_output_dir
        ]
        if args.crop:
            cmd_undistort.append("--crop")
        print(f"Running: {' '.join(cmd_undistort)}")
        try:
            subprocess.run(cmd_undistort, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] undistortionNerfstudioColmap.py failed for {folder_name}.")

    # Command 6: Run build_houdini_scene.py
    if not args.skip_houdini:
        houdini_script = os.path.join(script_dir, "build_houdini_scene.py")
        print("Scanning output directory for Houdini scene generation...")

        # Determine hython executable
        if args.hfs:
            # Clean up the path (remove quotes if any)
            hfs_path = args.hfs.strip().strip('"').strip("'")
            
            # Smart check for bin folder
            if os.path.basename(hfs_path).lower() == "bin":
                hython_dir = hfs_path
            else:
                hython_dir = os.path.join(hfs_path, "bin")
            
            hython_exec = os.path.join(hython_dir, "hython")
            if sys.platform == "win32":
                hython_exec += ".exe"
        else:
            hython_exec = "hython"
        
        subfolders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
        for folder in subfolders:
            folder_path = os.path.join(output_path, folder)
            ply_path = os.path.join(folder_path, "points3D.ply").replace("\\", "/")
            undistort_dir = os.path.join(folder_path, "undistort")
            json_path = os.path.join(undistort_dir, "transforms_undistorted.json").replace("\\", "/")
            hip_path = os.path.join(folder_path, f"{folder}.hip").replace("\\", "/")

            if os.path.exists(ply_path) and os.path.exists(json_path):
                # Ensure paths are absolutely correct for the OS
                h_exec = os.path.abspath(hython_exec)
                h_script = os.path.abspath(houdini_script)
                h_json = os.path.abspath(json_path)
                h_ply = os.path.abspath(ply_path)
                h_hip = os.path.abspath(hip_path)

                cmd_houdini = [h_exec, h_script, h_json, h_ply, h_hip,
                               "--sensor_width_mm", str(args.sensor_width_mm)]
                print(f"Running Houdini: {' '.join(cmd_houdini)}")
                
                # Check if executable exists and is not a directory
                if not os.path.isfile(h_exec):
                    print(f"[ERROR] Hython executable not found or is a directory: {h_exec}")
                    continue

                try:
                    # Clean environment to avoid Houdini picking up this script's Python venv
                    clean_env = os.environ.copy()
                    clean_env.pop("PYTHONPATH", None)
                    clean_env.pop("PYTHONHOME", None)
                    
                    # On Windows, sometimes shell=True helps with certain environment/path issues for Houdini
                    # but list-based is generally preferred. We'll stick to list but ensure absolute paths.
                    subprocess.run(cmd_houdini, check=True, env=clean_env)
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] build_houdini_scene.py failed for {folder} with exit code {e.returncode}.")
                except PermissionError:
                    print(f"[ERROR] Access denied when trying to run: {h_exec}")
                    print(f"        This might be due to file permissions, antivirus blocking, or {h_exec} being a directory.")
                except Exception as e:
                    print(f"[ERROR] An unexpected error occurred while running Houdini: {e}")
    else:
        print("Skipping Houdini scene generation as per argument.")

if __name__ == "__main__":
    main()