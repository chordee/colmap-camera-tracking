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
4. **Min observations / Max tracks** — optional thresholds, both default to
   **0 (no filter / no limit)**. With the defaults, every track in the
   Production Candidate Pool (see Structural QC below) is exported —
   nothing is sampled or dropped unless you explicitly set one of these.
   *Min observations* drops tracks that exist for fewer frames (a
   short-lived track is more likely a spurious COLMAP match). *Max tracks*
   randomly samples down to a cap, useful when a scene has far more tracks
   than are practical to import at once. Both are live-previewed against
   the **exportable** track count before you export.
5. **Generate candidate coverage pre-audit** — optional, default checked.
   May slow down export on large scenes (it does a full pass over every
   candidate track's observations). See Production Candidate Pool below.
6. **Export** — writes files to the output folder:
   - `<scene_name>_2d_tracks.txt` — the 3DE-native 2D Tracks file, ready to
     import.
   - `<scene_name>_structural_qc.txt` — a QC report (see below).
   - `<scene_name>_track_analysis.txt` and
     `<scene_name>_candidate_coverage_pre_selection.json` — written only if
     the pre-audit checkbox is checked (see Production Candidate Pool
     below).

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

This is the *only* automatic exclusion this tool performs. It exists
because the excluded tracks are structurally broken data (a same-frame
conflict, an out-of-bounds coordinate, a missing source image, an
exact-duplicate observation sequence) — not a judgment call about which
tracks are "good enough" to keep. Combined with the Min observations / Max
tracks defaults above, the tool's default behavior is to pass through
100% of the structurally valid, non-duplicate track set, with every
exclusion fully accounted for in the QC report.

## Production Candidate Pool and Coverage Pre-Audit

`candidate_pool.py` builds on the structural/duplicate classification above
to answer a different question: before any selection or budget logic runs
(that's future work), how much real candidate support does the scene
actually have, frame by frame and region by region?

- **Production Candidate Pool** — `build_production_candidate_pool` selects
  exactly the same tracks as `is_exportable` (structurally valid, non-
  conflict, non-duplicate). It's a separate, named entry point so future
  selection/budget logic has a stable pool to build on, without depending
  on the export-filter function by name.
- **Track Quality Analysis** — `analyze_track_quality` computes, per
  candidate track: observation count, first/last frame, temporal span,
  observation coverage (`observation_count / temporal_span`), and gap
  count/size. This is descriptive only — nothing here ever excludes a
  track from the pool or from export.
- **Candidate Coverage Pre-Audit** — `build_temporal_coverage_audit` and
  `build_spatial_coverage_audit` measure, across the whole candidate pool:
  how many candidate tracks are active per frame (with bucketed
  histograms, longest zero-coverage/low-support runs, and boundary
  minimums), and which regions of a 3x3 image-space grid ever have
  candidate support at all. This tells you whether a future selection
  stage's shortfalls come from thin upstream data or from the selection
  itself.

Both are written to disk by the GUI's pre-audit checkbox above, or
callable directly — see `write_track_analysis_report` and
`write_candidate_coverage_json`.

## Validation

Every change to this tool's export format has been verified by actually
importing the output into a real 3DEqualizer4 install, not just by
passing this project's own unit tests. This distinction matters: a file
that satisfies this project's own parser/writer round-trip tests can still
fail (or succeed with silently wrong geometry) when handed to the real
target software — this has happened during development here more than
once (a missing Y-axis flip that would have produced a mirrored-but-valid
import; a missing `setCurrentCamera`/`setCurrentPGroup` call that left an
imported camera with no visible animation despite importing without
error). Passing this project's tests means the parser-level contract is
met; it does not by itself mean a real 3DE import will look correct.

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
