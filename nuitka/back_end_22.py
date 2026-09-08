"""
Binary Signal Analyzer backend extension.

This module keeps the existing repository backend (`back_end_secure.py`) as the
analysis engine and owns all non-UI logic added for reference-binary matching
and PDF appendix generation.

Expected files in the same application directory:
    front_end_22.py
    back_end_22.py
    back_end_secure.py
"""

import os
import mmap
from dataclasses import dataclass
from pathlib import Path

try:
    import back_end_secure as _base
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "back_end_22.py requires the original back_end_secure.py in the same "
        "Python import path. Put back_end_secure.py next to front_end_22.py "
        "and back_end_22.py, or integrate the original backend into back_end_22.py."
    ) from exc

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, Spacer, Table, TableStyle


# Re-export the analysis API expected by the frontend.
REPORT_THEME = _base.REPORT_THEME
OUTPUT_DIR = _base.OUTPUT_DIR
IMG_DIR = _base.IMG_DIR
PDF_PATH = _base.PDF_PATH

load_bits = _base.load_bits
generate_plots = _base.generate_plots


def _sync_paths_to_base():
    """Keep output globals synchronized with the original backend."""
    _base.OUTPUT_DIR = OUTPUT_DIR
    _base.IMG_DIR = IMG_DIR
    _base.PDF_PATH = PDF_PATH


def setup_dirs():
    _sync_paths_to_base()
    _base.setup_dirs()


@dataclass
class ReferenceMatch:
    """Exact byte-search result. Offsets are zero-based and inclusive."""

    slot: int
    filename: str
    size: int
    offsets: list
    hex_preview: str = ""
    bit_preview: str = ""

    @property
    def found(self):
        return bool(self.offsets)


def find_all_byte_occurrences(data, pattern):
    """Return every exact byte occurrence, including overlapping occurrences."""
    if not pattern or len(pattern) > len(data):
        return []

    offsets = []
    position = data.find(pattern)
    while position >= 0:
        offsets.append(position)
        position = data.find(pattern, position + 1)
    return offsets


def search_reference_binaries(main_path, reference_paths):
    """Search complete reference contents in the main file using byte offsets."""
    main_size = os.path.getsize(main_path)
    results = []

    with open(main_path, "rb") as main_file:
        mapped = mmap.mmap(main_file.fileno(), 0, access=mmap.ACCESS_READ) if main_size else b""
        try:
            for slot, reference_path in enumerate(reference_paths, start=1):
                if not reference_path:
                    continue

                reference = Path(reference_path).read_bytes()
                hex_text = reference.hex(" ")
                bit_text = "".join(f"{byte:08b}" for byte in reference)

                if len(hex_text) > 512:
                    hex_text = hex_text[:240] + " ... " + hex_text[-240:]
                if len(bit_text) > 1024:
                    bit_text = bit_text[:480] + " ... " + bit_text[-480:]

                results.append(
                    ReferenceMatch(
                        slot=slot,
                        filename=Path(reference_path).name,
                        size=len(reference),
                        offsets=find_all_byte_occurrences(mapped, reference),
                        hex_preview=hex_text,
                        bit_preview=bit_text,
                    )
                )
        finally:
            if hasattr(mapped, "close"):
                mapped.close()

    return results


def _safe_text(value):
    from xml.sax.saxutils import escape
    return escape(str(value).encode("ascii", "backslashreplace").decode("ascii"))


def _appendix_table(rows, widths, header_color, font_size=8):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4D0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_original_report_with_appendix(metrics, paths, source_file, matches):
    """Build the original PDF and append the reference-matching pages."""
    _sync_paths_to_base()

    body = ParagraphStyle(
        "AppendixBody",
        fontName="Helvetica",
        fontSize=8.5,
        leading=10.5,
    )
    small = ParagraphStyle(
        "AppendixSmall",
        parent=body,
        fontSize=6.4,
        leading=7.4,
    )
    title = ParagraphStyle(
        "AppendixTitle",
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#163A5F"),
    )
    heading = ParagraphStyle(
        "AppendixHeading",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
    )
    subheading = ParagraphStyle(
        "AppendixSubheading",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
    )
    figure_heading = ParagraphStyle(
        "FigureHeading",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
    )
    header_color = colors.HexColor("#163A5F")

    appendix = []
    if matches:
        appendix.extend([
            PageBreak(),
            Paragraph("Appendix - Reference Binary Occurrences", title),
            Paragraph(
                "Reference contents are matched exactly as complete byte sequences. "
                "Offsets are zero-based bytes and both start and end offsets are inclusive. "
                "End offset = start offset + reference size - 1. The original report pages "
                "and plots remain unchanged.",
                body,
            ),
            Spacer(1, 0.25 * cm),
        ])

        for match in matches:
            found_text = "Yes" if match.found else ("Not searched" if match.size == 0 else "No")
            appendix.extend([
                Paragraph(f"Reference Binary {match.slot}: {_safe_text(match.filename)}", heading),
                _appendix_table([
                    ["Reference", "Size (bytes)", "Found", "Occurrences"],
                    [f"Reference {match.slot}", f"{match.size:,}", found_text, str(len(match.offsets))],
                ], [4.8 * cm, 3.2 * cm, 3.2 * cm, 4.4 * cm], header_color),
                Spacer(1, 0.12 * cm),
            ])

            if match.size == 0:
                appendix.append(Paragraph("The reference file is empty; no byte sequence was searched.", body))
            elif not match.found:
                appendix.append(Paragraph("No exact occurrence found.", body))
            else:
                rows = [["Occurrence", "Start", "End", "Start Hex", "End Hex"]]
                rows.extend([
                    [
                        str(index),
                        str(start),
                        str(start + match.size - 1),
                        f"0x{start:X}",
                        f"0x{start + match.size - 1:X}",
                    ]
                    for index, start in enumerate(match.offsets, 1)
                ])
                appendix.append(
                    _appendix_table(
                        rows,
                        [2.4 * cm, 3.1 * cm, 3.1 * cm, 3.2 * cm, 3.2 * cm],
                        header_color,
                    )
                )

            if match.size:
                appendix.extend([
                    Paragraph("Reference byte values (hexadecimal)", subheading),
                    Paragraph(_safe_text(match.hex_preview), small),
                    Paragraph("Reference bit values (8 bits per byte)", subheading),
                    Paragraph(_safe_text(match.bit_preview), small),
                ])
            appendix.append(Spacer(1, 0.28 * cm))

        appendix.extend([
            PageBreak(),
            Paragraph("Appendix - Binary Representation Graphics", title),
            Paragraph(
                "These figures are the same generated visualizations used by the original analysis report.",
                body,
            ),
            Spacer(1, 0.2 * cm),
        ])

        for key, caption in (
            ("signal", "Complete binary signal by sample index."),
            ("binary_2d", "Binary image representation of the analyzed bitstream."),
        ):
            path = paths.get(key, "")
            if path and os.path.isfile(path):
                appendix.extend([
                    Paragraph(caption, figure_heading),
                    Image(path, width=16 * cm, height=8 * cm),
                    Spacer(1, 0.25 * cm),
                ])

    # Preserve the original report builder. The interception happens entirely
    # inside the backend and is restored immediately after this single build.
    original_build = _base.SimpleDocTemplate.build

    def build_with_appendix(document, flowables, *args, **kwargs):
        if appendix:
            flowables.extend(appendix)
        return original_build(document, flowables, *args, **kwargs)

    _base.SimpleDocTemplate.build = build_with_appendix
    try:
        _base.build_pdf(metrics, paths, source_file=source_file)
    finally:
        _base.SimpleDocTemplate.build = original_build
