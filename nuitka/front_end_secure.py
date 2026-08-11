import os
import sys
import traceback
import ctypes

from PySide6.QtCore import QObject, QThread, Signal, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QComboBox,
    QProgressBar, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSizePolicy, QDialog, QTextBrowser
)



APP_USER_MODEL_ID = "BinarySignalAnalyzer.Desktop.1"
APP_ICON_FILE = "binary_analyzer.ico"


def configure_windows_app_identity():
    """Give the process a stable Windows taskbar identity."""
    if sys.platform != "win32":
        return

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID
        )
    except Exception:
        pass


def resource_path(filename):
    """Return a resource path that works in source and Nuitka bundles."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def load_application_icon():
    """Load the Qt application icon if the ICO file exists."""
    icon_path = resource_path(APP_ICON_FILE)
    if os.path.isfile(icon_path):
        return QIcon(icon_path)
    return QIcon()


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
QMenuBar {
    background: #0b1118;
    color: #eaf0f6;
    border-bottom: 1px solid #263647;
    padding: 3px 6px;
}
QMenuBar::item {
    background: transparent;
    padding: 5px 10px;
}
QMenuBar::item:selected {
    background: #223245;
}
QMenu {
    background: #151f2b;
    color: #eaf0f6;
    border: 1px solid #30465e;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 10px;
}
QMenu::item:selected {
    background: #2f7cf6;
    color: white;
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
        # Heavy analysis dependencies are imported only when the user starts an analysis.
        # This keeps application startup fast because NumPy, SciPy, Matplotlib,
        # PyWavelets and ReportLab are not loaded before the GUI appears.
        import back_end as analyzer

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
        except Exception as exc:
            # Do not propagate tracebacks or filesystem paths into the GUI.
            # The signal payload is intentionally generic.
            self.failed.emit(type(exc).__name__)


class AnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None

        # Real filesystem paths are private application state.
        # They are intentionally never displayed in the UI.
        self._input_path = ""
        self._output_path = os.path.abspath(
            os.path.join(os.getcwd(), "analysis_report")
        )
        self.pdf_path = ""

        self.setWindowTitle("Binary Signal Analyzer")
        self.create_menu_bar()
        self.setWindowIcon(load_application_icon())
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
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("No input file selected")
        browse_input = QPushButton("Browse…")
        browse_input.clicked.connect(self.choose_input)
        file_row.addWidget(self.input_edit, 1)
        file_row.addWidget(browse_input)
        input_layout.addLayout(file_row)

        input_layout.addWidget(self.field_label("Report folder"))
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit("Default report folder")
        self.output_edit.setReadOnly(True)
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

        self.path_label = QLabel("No report generated yet.")
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

    def create_menu_bar(self):
        """Create the only added UI: File | Help | Licences."""
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        menu_bar.setVisible(True)

        file_menu = menu_bar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menu_bar.addMenu("Help")
        manual_action = QAction("User Manual", self)
        manual_action.setShortcut("F1")
        manual_action.triggered.connect(self.show_user_manual)
        help_menu.addAction(manual_action)

        about_action = QAction("About Binary Signal Analyzer", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        licences_menu = menu_bar.addMenu("Licences")
        licences_action = QAction("Third-Party Licences", self)
        licences_action.triggered.connect(self.show_licences)
        licences_menu.addAction(licences_action)

    def show_user_manual(self):
        """Open the local PDF user manual."""
        manual_path = resource_path("user_manual.pdf")

        if not os.path.isfile(manual_path):
            QMessageBox.warning(
                self,
                "User Manual",
                "The local user manual is not available in this installation.",
            )
            return

        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(manual_path))
        if not opened:
            QMessageBox.warning(
                self,
                "User Manual",
                "The operating system could not open the PDF manual.",
            )

    def show_about(self):
        QMessageBox.about(
            self,
            "About Binary Signal Analyzer",
            """
            <h2>Binary Signal Analyzer</h2>
            <p>Offline binary signal analysis and PDF reporting application.</p>
            <p><b>Version:</b> 1.0</p>
            """,
        )

    def show_licences(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Binary Signal Analyzer - Third-Party Licences")
        dialog.setWindowIcon(load_application_icon())
        dialog.resize(900, 680)
        dialog.setMinimumSize(700, 500)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)

        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(False)
        browser.setHtml("""
        <style>
            body { font-family: "Segoe UI"; font-size: 10pt; color: #1f2937; background: white; }
            h1 { color: #17324d; }
            table { border-collapse: collapse; width: 100%; margin: 10px 0 18px 0; }
            th { background: #17324d; color: white; padding: 7px; text-align: left; }
            td { border: 1px solid #cfd8e3; padding: 7px; vertical-align: top; }
            .note { background: #eef4f8; border-left: 4px solid #2f6f9f; padding: 10px; margin: 12px 0; }
        </style>

        <h1>Third-Party Licences</h1>

        <table>
            <tr><th>Component</th><th>Purpose</th><th>Licence</th></tr>
            <tr><td><b>Python / CPython</b></td><td>Language and standard library</td><td>Python Software Foundation License</td></tr>
            <tr><td><b>PySide6 / Qt for Python</b></td><td>Desktop GUI</td><td>Community licensing includes LGPLv3 / GPLv3; commercial licensing also available</td></tr>
            <tr><td><b>Qt 6</b></td><td>GUI and platform runtime</td><td>Module-dependent; community modules commonly LGPLv3/GPLv3</td></tr>
            <tr><td><b>NumPy</b></td><td>Numerical processing</td><td>BSD-style licence</td></tr>
            <tr><td><b>SciPy</b></td><td>Scientific and signal-processing routines</td><td>BSD 3-Clause License</td></tr>
            <tr><td><b>PyWavelets</b></td><td>CWT and DWT</td><td>MIT License</td></tr>
            <tr><td><b>Matplotlib</b></td><td>Plot generation</td><td>Matplotlib licence; PSF-derived / BSD-compatible</td></tr>
            <tr><td><b>ReportLab</b></td><td>PDF generation</td><td>BSD License</td></tr>
            <tr><td><b>Nuitka</b></td><td>Optional executable compilation</td><td>Apache License 2.0 for the open-source compiler</td></tr>
            <tr><td><b>MinGW-w64 / GCC runtime</b></td><td>Optional native build toolchain</td><td>Component-specific GNU/free-software licences and applicable runtime exceptions</td></tr>
        </table>

        <div class="note">
        Preserve the complete licence files and notices for the exact dependency
        versions included in any released build.
        </div>
        """)

        layout.addWidget(browser)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close_button)
        layout.addLayout(row)

        dialog.exec()

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
            self._input_path = path
            self.input_edit.setText(os.path.basename(path))

    def choose_output(self):
        start_dir = self._output_path if os.path.isdir(self._output_path) else os.getcwd()

        path = QFileDialog.getExistingDirectory(
            self,
            "Select report folder",
            start_dir,
        )

        if path:
            self._output_path = path
            self.output_edit.setText("Custom report folder selected")

    def start_analysis(self):
        input_file = self._input_path
        output_dir = self._output_path

        if not input_file or not os.path.isfile(input_file):
            QMessageBox.warning(self, "Input file", "Please select a valid input file.")
            return
        if not output_dir:
            QMessageBox.warning(
                self,
                "Output folder",
                "Please select an output folder.",
            )
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
        self.path_label.setText("Report generated successfully.")
        self.run_button.setEnabled(True)
        self.open_pdf_button.setEnabled(True)
        self.open_folder_button.setEnabled(True)

    def analysis_failed(self, details):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label.setText("Analysis failed.")
        self.path_label.setText("No report was generated.")
        self.run_button.setEnabled(True)

        QMessageBox.critical(
            self,
            "Analysis error",
            "The analysis could not be completed. "
            "No local filesystem paths or internal traceback information "
            "are exposed by the application.",
        )

    def open_pdf(self):
        if self.pdf_path and os.path.isfile(self.pdf_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.pdf_path))

    def open_folder(self):
        if self.pdf_path:
            folder = os.path.dirname(self.pdf_path)
            if os.path.isdir(folder):
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))


def main():
    # Configure Windows identity before QApplication is created.
    configure_windows_app_identity()

    app = QApplication(sys.argv)
    app.setApplicationName("Binary Signal Analyzer")
    app.setApplicationDisplayName("Binary Signal Analyzer")
    app.setOrganizationName("Binary Signal Analyzer")
    app.setStyleSheet(APP_STYLE)
    app.setFont(QFont("Segoe UI", 10))

    app_icon = load_application_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    window = AnalyzerWindow()
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
