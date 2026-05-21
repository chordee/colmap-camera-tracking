import os
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_AUTOTRACKER = SCRIPT_DIR / "run_autotracker.py"


class PathPicker(QWidget):
    """Line edit + browse button for selecting a file or directory."""

    def __init__(self, mode="dir", file_filter="All files (*.*)", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.file_filter = file_filter

        self.edit = QLineEdit()
        self.button = QPushButton("Browse…")
        self.button.clicked.connect(self._browse)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)

    def _browse(self):
        if self.mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select directory", self.edit.text() or str(SCRIPT_DIR))
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", self.edit.text() or str(SCRIPT_DIR), self.file_filter)
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("COLMAP Camera Tracking")
        self.resize(900, 700)

        self.process: QProcess | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        root.addWidget(self._build_io_group())
        root.addWidget(self._build_tabs(), stretch=1)
        root.addWidget(self._build_buttons())
        root.addWidget(self._build_log(), stretch=2)

    # ------------------------------------------------------------------ UI

    def _build_io_group(self) -> QGroupBox:
        group = QGroupBox("Input / Output")
        form = QFormLayout(group)

        self.input_picker = PathPicker(mode="dir")
        self.output_picker = PathPicker(mode="dir")

        form.addRow("Input videos dir:", self.input_picker)
        form.addRow("Output dir:", self.output_picker)
        return group

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._tab_basic(), "Basic")
        tabs.addTab(self._tab_camera(), "Camera")
        tabs.addTab(self._tab_color_mask(), "Color / Mask")
        tabs.addTab(self._tab_loop(), "Loop Detection")
        tabs.addTab(self._tab_houdini(), "Houdini")
        return tabs

    def _tab_basic(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.1, 1.0)
        self.scale.setSingleStep(0.05)
        self.scale.setDecimals(2)
        self.scale.setValue(0.5)
        form.addRow("Scale:", self.scale)

        self.overlap = QSpinBox()
        self.overlap.setRange(1, 200)
        self.overlap.setValue(12)
        form.addRow("Sequential overlap:", self.overlap)

        self.crop = QCheckBox("Keep original canvas size (skip expansion)")
        form.addRow(self.crop)

        self.camera_model = QComboBox()
        self.camera_model.addItems([
            "(auto)",
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "SIMPLE_RADIAL",
            "RADIAL",
            "OPENCV",
            "OPENCV_FISHEYE",
        ])
        form.addRow("Camera model:", self.camera_model)

        return w

    def _tab_camera(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.focal_length_mm = QDoubleSpinBox()
        self.focal_length_mm.setRange(0.0, 1000.0)
        self.focal_length_mm.setSingleStep(0.5)
        self.focal_length_mm.setDecimals(2)
        self.focal_length_mm.setSpecialValueText("(auto)")
        self.focal_length_mm.setValue(0.0)
        form.addRow("Focal length (mm):", self.focal_length_mm)

        self.sensor_width_mm = QDoubleSpinBox()
        self.sensor_width_mm.setRange(1.0, 100.0)
        self.sensor_width_mm.setSingleStep(0.1)
        self.sensor_width_mm.setDecimals(2)
        self.sensor_width_mm.setValue(36.0)
        form.addRow("Sensor width (mm):", self.sensor_width_mm)

        hint = QLabel(
            "Common widths: Full Frame=36.0, ARRI LF=36.7, Super35=24.89, MFT=17.3"
        )
        hint.setStyleSheet("color: gray;")
        form.addRow(hint)

        return w

    def _tab_color_mask(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.acescg = QCheckBox("Convert ACEScg → sRGB before processing")
        form.addRow(self.acescg)

        self.lut = PathPicker(mode="file", file_filter="LUT (*.cube)")
        form.addRow("LUT (.cube):", self.lut)

        self.mask = PathPicker(mode="dir")
        form.addRow("Mask root dir:", self.mask)

        return w

    def _tab_loop(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.loop = QCheckBox("Enable loop detection")
        form.addRow(self.loop)

        self.loop_period = QSpinBox()
        self.loop_period.setRange(1, 100)
        self.loop_period.setValue(5)
        form.addRow("Period:", self.loop_period)

        self.loop_num_images = QSpinBox()
        self.loop_num_images.setRange(1, 1000)
        self.loop_num_images.setValue(50)
        form.addRow("Images per pass:", self.loop_num_images)

        self.vocab_tree = PathPicker(mode="file", file_filter="Vocab tree (*.bin)")
        default_vocab = SCRIPT_DIR / "vocab_tree_faiss_flickr100K_words32K.bin"
        if default_vocab.exists():
            self.vocab_tree.edit.setText(str(default_vocab))
        form.addRow("Vocab tree:", self.vocab_tree)

        return w

    def _tab_houdini(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.skip_houdini = QCheckBox("Skip Houdini scene generation")
        form.addRow(self.skip_houdini)

        self.hfs = PathPicker(mode="dir")
        form.addRow("Houdini install (HFS):", self.hfs)

        self.multi_cams = QCheckBox("Treat each video as a separate camera (multi-cams)")
        form.addRow(self.multi_cams)

        return w

    def _build_buttons(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.clear_log_btn = QPushButton("Clear log")
        self.clear_log_btn.clicked.connect(lambda: self.log.clear())

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addStretch()
        layout.addWidget(self.clear_log_btn)
        return container

    def _build_log(self) -> QPlainTextEdit:
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        self.log.setFont(font)
        return self.log

    # -------------------------------------------------------------- helpers

    def _append_log(self, text: str):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def _build_command(self) -> list[str] | None:
        input_dir = self.input_picker.text()
        output_dir = self.output_picker.text()

        if not input_dir or not Path(input_dir).is_dir():
            QMessageBox.warning(self, "Missing input", "Please choose a valid input videos directory.")
            return None
        if not output_dir:
            QMessageBox.warning(self, "Missing output", "Please choose an output directory.")
            return None

        cmd = [sys.executable, str(RUN_AUTOTRACKER), input_dir, output_dir]

        cmd += ["--scale", f"{self.scale.value():.4f}"]
        cmd += ["--overlap", str(self.overlap.value())]

        if self.crop.isChecked():
            cmd.append("--crop")

        if self.camera_model.currentText() != "(auto)":
            cmd += ["--camera_model", self.camera_model.currentText()]

        if self.focal_length_mm.value() > 0:
            cmd += ["--focal_length_mm", f"{self.focal_length_mm.value():.4f}"]
            cmd += ["--sensor_width_mm", f"{self.sensor_width_mm.value():.4f}"]
        elif self.sensor_width_mm.value() != 36.0:
            cmd += ["--sensor_width_mm", f"{self.sensor_width_mm.value():.4f}"]

        if self.acescg.isChecked():
            cmd.append("--acescg")
        if not self._append_path(cmd, "--lut", self.lut.text(), "LUT file", expect_dir=False):
            return None
        if not self._append_path(cmd, "--mask", self.mask.text(), "Mask directory", expect_dir=True):
            return None

        if self.loop.isChecked():
            cmd.append("--loop")
            cmd += ["--loop_period", str(self.loop_period.value())]
            cmd += ["--loop_num_images", str(self.loop_num_images.value())]
            if not self._append_path(cmd, "--vocab_tree_path", self.vocab_tree.text(), "Vocab tree", expect_dir=False):
                return None

        if self.skip_houdini.isChecked():
            cmd.append("--skip-houdini")
        if not self._append_path(cmd, "--hfs", self.hfs.text(), "Houdini install (HFS)", expect_dir=True):
            return None
        if self.multi_cams.isChecked():
            cmd.append("--multi-cams")

        return cmd

    def _append_path(self, cmd: list[str], flag: str, value: str, label: str, expect_dir: bool) -> bool:
        """Append `flag value` to cmd if value is non-empty and exists.

        Returns False (with a QMessageBox warning) if the value is non-empty but
        the path does not exist or has the wrong type. Empty values are skipped.
        """
        if not value:
            return True
        path = Path(value)
        if not path.exists():
            QMessageBox.warning(self, "Invalid path", f"{label} does not exist:\n{value}")
            return False
        if expect_dir and not path.is_dir():
            QMessageBox.warning(self, "Invalid path", f"{label} is not a directory:\n{value}")
            return False
        if not expect_dir and not path.is_file():
            QMessageBox.warning(self, "Invalid path", f"{label} is not a file:\n{value}")
            return False
        cmd += [flag, value]
        return True

    # ------------------------------------------------------------- actions

    def _start(self):
        if self.process is not None:
            return

        cmd = self._build_command()
        if cmd is None:
            return

        if not RUN_AUTOTRACKER.exists():
            QMessageBox.critical(self, "Missing script", f"Cannot find {RUN_AUTOTRACKER}.")
            return

        self._append_log("\n$ " + " ".join(cmd) + "\n")

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)

        env = self.process.processEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.process.start(cmd[0], cmd[1:])

    def _stop(self):
        if self.process is None:
            return
        self.process.terminate()
        QTimer.singleShot(3000, self._kill_if_still_running)

    def _kill_if_still_running(self):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def _on_output(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_log(data)

    def _on_finished(self, exit_code: int, exit_status):
        self._append_log(f"\n[Process exited with code {exit_code}]\n")
        self.process = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_error(self, error):
        self._append_log(f"\n[Process error: {error}]\n")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
