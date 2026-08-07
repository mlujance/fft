import os
import sys
import traceback

from PySide6.QtCore import QObject, QThread, Signal, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QComboBox,
    QProgressBar, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSizePolicy
)

import binary_analysis_pdf_compact as analyzer


APP_STYLE = """
QWidget {
    background: #0f1720;
    color: #eaf0f6;
    font-family: "Segoe UI";
    font-size: 10.5pt;
}
QMainWindow {
    background: #0b1118;
}
QFrame#Card {
    background: #151f2b;
    border: 1px solid #263647;
    border-radius: 14px;
}
QLabel#Title {
    font-size: 24pt;
    font-weight: 700;
    color: #f7fafc;
}
QLabel#Subtitle {
    color: #8fa3b8;
    font-size: 10.5pt;
}
QLabel#SectionTitle {
    font-size: 12pt;
    font-weight: 650;
    color: #f3f7fb;
}
QLabel#FieldLabel {
    color: #aebdcb;
    font-size: 9.5pt;
}
QLabel#PathLabel {
    color: #9fb0c0;
    background: #0d151e;
    border: 1px solid #263647;
    border-radius: 9px;
    padding: 10px 12px;
}
QLineEdit, QDoubleSpinBox, QComboBox {
    background: #0d151e;
    border: 1px solid #2a3b4d;
    border-radius: 9px;
    padding: 9px 11px;
    color: #eef5fb;
    selection-background-color: #2f7cf6;
}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #4a90ff;
}
QPushButton {
    background: #223245;
    border: 1px solid #30465e;
    border-radius: 9px;
    padding: 9px 15px;
    font-weight: 600;
}
QPushButton:hover {
    background: #2a3e55;
}
QPushButton:pressed {
    background: #1d2b3a;
}
QPushButton#PrimaryButton {
    background: #2f7cf6;
    border: 1px solid #2f7cf6;
    color: white;
    padding: 12px 18px;
    font-size: 11pt;
}
QPushButton#PrimaryButton:hover {
    background: #438cff;
}
QPushButton#SuccessButton {
    background: #163f34;
    border: 1px solid #235e4f;
    color: #d7fff3;
}
QPushButton:disabled {
    color: #6d7a87;
    background: #18222c;
    border-color: #24313e;
}
QProgressBar {
    background: #0d151e;
    border: 1px solid #263647;
    border-radius: 7px;
    min-height: 13px;
    max-height: 13px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: #2f7cf6;
    border-radius: 6px;
}
"""


class AnalysisWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, input_file, output_dir, fs, cwt_wavelet, dwt_wavelet):
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.fs = fs
        self.cwt_wavelet = cwt_wavelet
        self.dwt_wavelet = dwt_wavelet

    def run(self):
        try:
            output_dir = os.path.abspath(self.output_dir)
            images_dir = os.path.join(output_dir, "images")
            pdf_path = os.path.join(output_dir, "binary_analysis_report.pdf")

            analyzer.OUTPUT_DIR = output_dir
            analyzer.IMG_DIR = images_dir
            analyzer.PDF_PATH = pdf_path

            self.status.emit("Preparing output folders…")
            analyzer.setup_dirs()

            self.status.emit("Loading binary data…")
            bits = analyzer.load_bits(self.input_file)

            self.status.emit("Running spectral, statistical and multiscale analysis…")
            metrics, paths = analyzer.generate_plots(
                bits,
                fs=self.fs,
                cwt_wavelet=self.cwt_wavelet,
                dwt_wavelet=self.dwt_wavelet,
            )

            self.status.emit("Building PDF report…")
            analyzer.build_pdf(metrics, paths, source_file=self.input_file)

            self.finished.emit(os.path.abspath(pdf_path))
        except Exception:
            self.failed.emit(traceback.format_exc())


class AnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.pdf_path = ""

        self.setWindowTitle("Binary Signal Analyzer")
        self.setMinimumSize(900, 650)
        self.resize(980, 720)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(34, 30, 34, 30)
        outer.setSpacing(18)

        title = QLabel("Binary Signal Analyzer")
        title.setObjectName("Title")
        subtitle = QLabel("Professional multiscale binary analysis with automated PDF reporting")
        subtitle.setObjectName("Subtitle")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        input_card = self.make_card()
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(22, 20, 22, 20)
        input_layout.setSpacing(13)

        section = QLabel("Input & output")
        section.setObjectName("SectionTitle")
        input_layout.addWidget(section)

        input_layout.addWidget(self.field_label("Input file"))
        file_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select a .txt, .csv or binary file")
        browse_input = QPushButton("Browse…")
        browse_input.clicked.connect(self.choose_input)
        file_row.addWidget(self.input_edit, 1)
        file_row.addWidget(browse_input)
        input_layout.addLayout(file_row)

        input_layout.addWidget(self.field_label("Report folder"))
        output_row = QHBoxLayout()
        default_output = os.path.abspath(os.path.join(os.getcwd(), "analysis_report"))
        self.output_edit = QLineEdit(default_output)
        browse_output = QPushButton("Choose folder…")
        browse_output.clicked.connect(self.choose_output)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse_output)
        input_layout.addLayout(output_row)

        outer.addWidget(input_card)

        settings_card = self.make_card()
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(22, 20, 22, 20)
        settings_layout.setSpacing(13)

        section2 = QLabel("Analysis settings")
        section2.setObjectName("SectionTitle")
        settings_layout.addWidget(section2)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)

        fs_label = self.field_label("Sampling frequency")
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(0.000001, 1_000_000_000.0)
        self.fs_spin.setDecimals(6)
        self.fs_spin.setValue(1.0)

        cwt_label = self.field_label("CWT wavelet")
        self.cwt_combo = QComboBox()
        self.cwt_combo.addItems(["morl", "mexh", "gaus1"])

        dwt_label = self.field_label("DWT wavelet")
        self.dwt_combo = QComboBox()
        self.dwt_combo.addItems(["db4", "db2", "haar", "sym4", "coif1"])

        grid.addWidget(fs_label, 0, 0)
        grid.addWidget(cwt_label, 0, 1)
        grid.addWidget(dwt_label, 0, 2)
        grid.addWidget(self.fs_spin, 1, 0)
        grid.addWidget(self.cwt_combo, 1, 1)
        grid.addWidget(self.dwt_combo, 1, 2)
        settings_layout.addLayout(grid)
        outer.addWidget(settings_card)

        action_row = QHBoxLayout()
        self.run_button = QPushButton("Run analysis")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self.start_analysis)
        action_row.addStretch(1)
        action_row.addWidget(self.run_button)
        outer.addLayout(action_row)

        result_card = self.make_card()
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(22, 20, 22, 20)
        result_layout.setSpacing(12)

        result_title = QLabel("Report status")
        result_title.setObjectName("SectionTitle")
        result_layout.addWidget(result_title)

        self.status_label = QLabel("Ready. Select a file and run the analysis.")
        self.status_label.setObjectName("Subtitle")
        result_layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        result_layout.addWidget(self.progress)

        self.path_label = QLabel("PDF path will appear here after generation.")
        self.path_label.setObjectName("PathLabel")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setWordWrap(True)
        result_layout.addWidget(self.path_label)

        buttons = QHBoxLayout()
        self.open_pdf_button = QPushButton("Open PDF")
        self.open_pdf_button.setObjectName("SuccessButton")
        self.open_pdf_button.setEnabled(False)
        self.open_pdf_button.clicked.connect(self.open_pdf)

        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self.open_folder)

        buttons.addWidget(self.open_pdf_button)
        buttons.addWidget(self.open_folder_button)
        buttons.addStretch(1)
        result_layout.addLayout(buttons)

        outer.addWidget(result_card)
        outer.addStretch(1)

    @staticmethod
    def make_card():
        card = QFrame()
        card.setObjectName("Card")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        return card

    @staticmethod
    def field_label(text):
        label = QLabel(text)
        label.setObjectName("FieldLabel")
        return label

    def choose_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input file",
            "",
            "Supported files (*.txt *.csv *.bin *.dat);;All files (*.*)",
        )
        if path:
            self.input_edit.setText(path)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select report folder",
            self.output_edit.text() or os.getcwd(),
        )
        if path:
            self.output_edit.setText(path)

    def start_analysis(self):
        input_file = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        if not input_file or not os.path.isfile(input_file):
            QMessageBox.warning(self, "Input file", "Please select a valid input file.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Output folder", "Please select an output folder.")
            return

        self.pdf_path = ""
        self.open_pdf_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.status_label.setText("Starting analysis…")
        self.path_label.setText("Generating report…")

        self.thread = QThread(self)
        self.worker = AnalysisWorker(
            input_file=input_file,
            output_dir=output_dir,
            fs=self.fs_spin.value(),
            cwt_wavelet=self.cwt_combo.currentText(),
            dwt_wavelet=self.dwt_combo.currentText(),
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def analysis_finished(self, pdf_path):
        self.pdf_path = pdf_path
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status_label.setText("Analysis complete. PDF report generated successfully.")
        self.path_label.setText(pdf_path)
        self.run_button.setEnabled(True)
        self.open_pdf_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)

    def analysis_failed(self, details):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label.setText("Analysis failed.")
        self.path_label.setText("No PDF was generated.")
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "Analysis error", details)

    def open_pdf(self):
        if self.pdf_path and os.path.isfile(self.pdf_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.pdf_path))

    def open_folder(self):
        if self.pdf_path:
            folder = os.path.dirname(self.pdf_path)
            if os.path.isdir(folder):
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Binary Signal Analyzer")
    app.setStyleSheet(APP_STYLE)
    app.setFont(QFont("Segoe UI", 10))

    window = AnalyzerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
