import os
import sys
import traceback

from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget,
    QSpinBox, QFormLayout,
)

from colmap_2d_tracks import (
    SceneLoadError, filter_by_min_observations, load_scene, sample_tracks,
    write_3de_2d_tracks_txt,
)


class PathPicker(QWidget):
    def __init__(self, mode="dir", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.edit = QLineEdit()
        self.button = QPushButton("Browse...")
        self.button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)

    def _browse(self):
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if path:
            self.edit.setText(path)

    def text(self):
        return self.edit.text().strip()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COLMAP -> 3DEqualizer 2D Track Export")
        self.resize(640, 480)
        self._scene_tracks = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        form = QFormLayout()
        self.scene_dir = PathPicker(mode="dir")
        self.scene_dir.edit.textChanged.connect(self._on_scene_dir_changed)
        form.addRow("Scene folder:", self.scene_dir)

        self.tracks_available = QLabel("Tracks available: (select a scene folder)")
        form.addRow("", self.tracks_available)

        self.output_dir = PathPicker(mode="dir")
        form.addRow("Output folder:", self.output_dir)

        self.production_start_frame = QSpinBox()
        self.production_start_frame.setRange(1, 1_000_000)
        self.production_start_frame.setValue(1)
        form.addRow("Production start frame:", self.production_start_frame)

        self.min_observations = QSpinBox()
        self.min_observations.setRange(0, 1_000_000)
        self.min_observations.setValue(0)
        self.min_observations.setSpecialValueText("No filter")
        self.min_observations.valueChanged.connect(self._update_tracks_preview)
        form.addRow("Min observations (0 = no filter):", self.min_observations)

        self.max_tracks = QSpinBox()
        self.max_tracks.setRange(0, 100_000_000)
        self.max_tracks.setValue(0)
        self.max_tracks.setSpecialValueText("No limit")
        form.addRow("Max tracks (0 = no limit):", self.max_tracks)

        layout.addLayout(form)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self._on_export)
        layout.addWidget(self.export_button)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def _log(self, message):
        self.log.append(message)

    def _on_scene_dir_changed(self, scene_dir):
        scene_dir = scene_dir.strip()
        if not scene_dir:
            self._scene_tracks = None
            self.tracks_available.setText("Tracks available: (select a scene folder)")
            return
        try:
            scene = load_scene(scene_dir)
        except SceneLoadError as e:
            self._scene_tracks = None
            self.tracks_available.setText(f"Tracks available: (could not read scene -- {e})")
            return
        except Exception:
            self._scene_tracks = None
            self.tracks_available.setText("Tracks available: (could not read scene)")
            return
        self._scene_tracks = scene["tracks"]
        if scene["conflict_count"]:
            self._log(f"[WARN] {scene['conflict_count']} track(s) dropped due to "
                       f"same-frame conflicts.")
        self._update_tracks_preview()

    def _update_tracks_preview(self):
        if self._scene_tracks is None:
            return
        total = len(self._scene_tracks)
        min_observations = self.min_observations.value()
        remaining = len(filter_by_min_observations(self._scene_tracks, min_observations))
        if min_observations:
            self.tracks_available.setText(
                f"Tracks available: {remaining} of {total} (observations >= {min_observations})"
            )
        else:
            self.tracks_available.setText(f"Tracks available: {total}")

    def _on_export(self):
        scene_dir = self.scene_dir.text()
        output_dir = self.output_dir.text()
        production_start_frame = self.production_start_frame.value()

        if not scene_dir:
            self._log("[ERROR] Please select a scene folder.")
            return
        if not output_dir:
            self._log("[ERROR] Please select an output folder.")
            return

        scene_name = os.path.basename(os.path.normpath(scene_dir))

        try:
            self._log(f"Loading scene: {scene_dir}")
            scene = load_scene(scene_dir)
        except SceneLoadError as e:
            self._log(f"[ERROR] {e}")
            return
        except Exception:
            self._log("[ERROR] Unexpected error while loading scene:")
            self._log(traceback.format_exc())
            return

        min_observations = self.min_observations.value()
        max_tracks = self.max_tracks.value()
        filtered_tracks = filter_by_min_observations(scene["tracks"], min_observations)
        tracks_to_export = sample_tracks(filtered_tracks, max_tracks)

        os.makedirs(output_dir, exist_ok=True)
        tracks_path = os.path.join(output_dir, f"{scene_name}_2d_tracks.txt")

        try:
            write_3de_2d_tracks_txt(tracks_to_export, production_start_frame, tracks_path)
        except Exception:
            self._log("[ERROR] Unexpected error while writing export file:")
            self._log(traceback.format_exc())
            return

        if min_observations:
            self._log(f"Kept {len(filtered_tracks)} of {len(scene['tracks'])} tracks "
                       f"with observations >= {min_observations}.")
        if len(tracks_to_export) < len(filtered_tracks):
            self._log(f"Exported {len(tracks_to_export)} of {len(filtered_tracks)} tracks "
                       f"(randomly sampled) to: {tracks_path}")
        else:
            self._log(f"Exported {len(tracks_to_export)} tracks to: {tracks_path}")
        self._log("")
        self._log("Next steps:")
        self._log("1. In 3DE, use the 2D Tracks import feature (Object Browser or "
                   "File > Import, depending on your 3DE R5 setup) to import this file.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
