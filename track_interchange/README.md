# track_interchange

Export COLMAP's raw 2D feature tracks (from a processed COLMAP sparse
reconstruction) to 3DEqualizer4's native "2D Tracks" ASCII format.

This is a separate, parallel tool from the 3D-camera/point-cloud exporter
used elsewhere in this repo. It does not touch COLMAP's solved 3D camera
path or point cloud — it re-derives *persistent 2D tracks* directly from
`images.txt`'s per-image 2D observations, keyed by COLMAP's `POINT3D_ID`,
so a 3DE artist can re-track/re-solve from the original 2D data inside
3DEqualizer instead of trusting COLMAP's 3D solve as-is.

## Requirements

- The repo's existing `uv`-managed Python environment (same one used by
  `run_autotracker.py`/`gui_autotracker.py`). No separate environment.
- `PySide6` (already part of the project's dependencies) if you're using
  the GUI.
- A processed COLMAP scene folder containing `sparse/0/cameras.txt` and
  `sparse/0/images.txt` (COLMAP's text export format, not the binary one).

## GUI usage

From the repo root:

```
uv run track_interchange/gui_2d_track_export.py
```

1. **Scene folder** — pick the processed COLMAP scene directory (the one
   containing `sparse/0/cameras.txt` and `sparse/0/images.txt`). The
   "Tracks available" label updates live once a valid scene is selected.
2. **Output folder** — where the exported files will be written.
3. **Production start frame** — the frame number your 3DE shot's first
   frame corresponds to (used to remap COLMAP's frame numbers, parsed from
   image filenames, onto 3DE's 1-based sequence frame numbers).
4. **Min observations / Max tracks** — optional thresholds. *Min
   observations* drops tracks that exist for fewer frames (a short-lived
   track is more likely a spurious COLMAP match). *Max tracks* randomly
   samples down to a cap, useful when a scene has far more tracks than are
   practical to import at once. Both are live-previewed against the
   **exportable** track count (see Structural QC below) before you export.
5. **Export** — writes two files to the output folder:
   - `<scene_name>_2d_tracks.txt` — the 3DE-native 2D Tracks file, ready to
     import.
   - `<scene_name>_structural_qc.txt` — a QC report (see below).

## Structural QC and Exact Duplicate filtering

Not every 2D observation COLMAP records is safe to hand to 3DE as-is. Before
export, every track is classified and only clean tracks are written out:

- **structural_status** — `VALID`, or one of:
  - `CONFLICT` — the same track has two different 2D positions in the same
    frame (a same-frame data conflict in `images.txt`).
  - `INVALID_COORDS` — a non-finite or out-of-image-bounds coordinate.
  - `INVALID_IMAGE_MISSING` — the frame's image file isn't present in the
    scene's `images/` folder. (If the scene folder has no `images/`
    directory at all, this specific check is skipped for every track,
    since that's expected for a scene where footage wasn't extracted
    alongside the reconstruction — real, complete scene folders always
    have this directory.)
  - `INVALID_FRAME` — the image filename had no parseable frame number.
- **duplicate_status** — `UNIQUE`, or `EXACT_DUPLICATE` if another track has
  an identical observation sequence (exact match, no tolerance). The
  lowest-numbered `POINT3D_ID` in a duplicate group is kept as `UNIQUE`;
  the rest are tagged as duplicates of it.

A track is only ever written to `<scene_name>_2d_tracks.txt` if
`structural_status == "VALID"` and `duplicate_status == "UNIQUE"`. Every
track that fails either check is retained internally (never silently
dropped) and listed by name in `<scene_name>_structural_qc.txt`, along with
summary counts, so you can see exactly what was excluded and why.

## Using `colmap_2d_tracks.py` directly

The GUI is a thin wrapper. To script the conversion yourself (batch jobs,
custom filtering, etc.), the core functions are in `colmap_2d_tracks.py`:

```python
# run from the repo root, so track_interchange resolves as a namespace package
from track_interchange.colmap_2d_tracks import (
    load_scene, is_exportable, filter_by_min_observations,
    sample_tracks, write_3de_2d_tracks_txt, write_structural_qc_report,
)

scene = load_scene("/path/to/colmap/scene")   # {"width", "height", "tracks"}

exportable = [t for t in scene["tracks"] if is_exportable(t)]
tracks = filter_by_min_observations(exportable, min_observations=5)
tracks = sample_tracks(tracks, max_tracks=20000)

write_3de_2d_tracks_txt(
    tracks, production_start_frame=1001,
    image_height=scene["height"], out_path="out/shot_2d_tracks.txt",
)
write_structural_qc_report(scene["tracks"], out_path="out/shot_structural_qc.txt")
```

Each track in `scene["tracks"]` is a dict:

```python
{
    "track_id": "colmap::<point3d_id>",
    "track_name": "p<point3d_id>",
    "observations": [
        {"production_frame": int, "x": float, "y": float, "image_name": str},
        ...
    ],
    "structural_status": "VALID" | "CONFLICT" | "INVALID_COORDS" |
                          "INVALID_IMAGE_MISSING" | "INVALID_FRAME",
    "duplicate_status": "UNIQUE" | "EXACT_DUPLICATE",
    "duplicate_of": str | None,
}
```

Coordinates in `observations` are in COLMAP's convention (top-left origin,
Y-down, pixel space). `write_3de_2d_tracks_txt` converts to 3DE's native
convention (bottom-left origin, Y-up) internally — you don't need to flip
anything yourself.

## Importing into 3DEqualizer

In 3DE, use the native **2D Tracks** import (Object Browser, or File >
Import, depending on your 3DE version) and point it at
`<scene_name>_2d_tracks.txt`. This is 3DE's own bundled ASCII format, not a
custom one — no plugin required.

## Tests

```
uv run python -m unittest discover -s track_interchange/tests -v
```

Pure `unittest`, stdlib only — no COLMAP or 3DE installation required to run
the test suite.
