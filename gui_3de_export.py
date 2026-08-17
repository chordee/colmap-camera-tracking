import os
import sys
import traceback

from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget,
    QDoubleSpinBox, QFormLayout,
)

from colmap_to_3de import SceneLoadError, load_scene, write_camera_import_script, write_survey_points_txt


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
        self.setWindowTitle("COLMAP -> 3DEqualizer Export")
        self.resize(640, 480)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        form = QFormLayout()
        self.scene_dir = PathPicker(mode="dir")
        form.addRow("Scene folder:", self.scene_dir)
        self.output_dir = PathPicker(mode="dir")
        form.addRow("Output folder:", self.output_dir)

        self.sensor_width_mm = QDoubleSpinBox()
        self.sensor_width_mm.setRange(1.0, 100.0)
        self.sensor_width_mm.setDecimals(2)
        self.sensor_width_mm.setValue(36.0)
        form.addRow("Sensor width (mm):", self.sensor_width_mm)

        layout.addLayout(form)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self._on_export)
        layout.addWidget(self.export_button)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def _log(self, message):
        self.log.append(message)

    def _on_export(self):
        scene_dir = self.scene_dir.text()
        output_dir = self.output_dir.text()
        sensor_width_mm = self.sensor_width_mm.value()

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

        os.makedirs(output_dir, exist_ok=True)
        points_path = os.path.join(output_dir, f"{scene_name}_points.txt")
        camera_script_path = os.path.join(output_dir, f"{scene_name}_import_camera.py")

        try:
            write_survey_points_txt(scene["points"], points_path)
            write_camera_import_script(scene, scene_name, sensor_width_mm, camera_script_path)
        except Exception:
            self._log("[ERROR] Unexpected error while writing export files:")
            self._log(traceback.format_exc())
            return

        self._log(f"Exported {len(scene['points'])} points to: {points_path}")
        self._log(f"Exported {len(scene['frames'])} frames to: {camera_script_path}")
        self._log("")
        self._log("Next steps:")
        self._log("1. Copy the _import_camera.py file into 3DE's Script Database "
                   "folder, then run it from 3DE's File > Import menu.")
        self._log("2. In 3DE, use File > Import > Import Survey Textfile... "
                   "to import the _points.txt file.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
