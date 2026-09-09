"""
Binary Signal Analyzer - compatibility backend.

This backend deliberately preserves the original analysis engine and the
original PDF/report aesthetics from back_end_secure.py. It only adds the
optional reference appendix required by front_end_22.py.

Important:
- All original metrics and plots are produced by back_end_secure.py unchanged.
- CWT/DWT plots therefore retain the exact original appearance and colormap.
- Reference comparison is performed BIT BY BIT, not byte by byte.
- Main and reference files are normalized through the SAME original load_bits()
  function. Therefore equivalent logical bitstreams can match even when one is
  stored as TXT/CSV and the other as BIN/DAT.
"""

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import back_end_secure as _base

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle,
)


# ---------------------------------------------------------------------------
# Re-export the original backend API and configuration.
# ---------------------------------------------------------------------------

REPORT_THEME = _base.REPORT_THEME

OUTPUT_DIR = _base.OUTPUT_DIR
IMG_DIR = _base.IMG_DIR
PDF_PATH = _base.PDF_PATH

load_bits = _base.load_bits
generate_plots = _base.generate_plots


def _sync_paths_to_base():
    """Synchronize paths configured by the frontend with the original backend."""
    _base.OUTPUT_DIR = OUTPUT_DIR
    _base.IMG_DIR = IMG_DIR
    _base.PDF_PATH = PDF_PATH


def setup_dirs():
    _sync_paths_to_base()
    _base.setup_dirs()


# ---------------------------------------------------------------------------
# Exact bit-level reference matching
# ---------------------------------------------------------------------------

@dataclass
class ReferenceMatch:
    slot: int
    filename: str
    bits: np.ndarray
    offsets: list

    @property
    def size_bits(self):
        return int(self.bits.size)

    @property
    def found(self):
        return bool(self.offsets)


def find_exact_bit_occurrences(main_bits, reference_bits):
    """
    Return every overlapping exact occurrence of reference_bits in main_bits.

    Both inputs are interpreted as normalized bit arrays. Positions are
    zero-based BIT offsets and the matching is exact.
    """
    main_bits = np.asarray(main_bits, dtype=np.uint8).ravel()
    reference_bits = np.asarray(reference_bits, dtype=np.uint8).ravel()

    main_len = int(main_bits.size)
    ref_len = int(reference_bits.size)

    if ref_len == 0 or ref_len > main_len:
        return []

    # Candidate filtering on the first bit avoids comparing the reference
    # against every single position while preserving exact overlapping matches.
    candidate_limit = main_len - ref_len + 1
    candidates = np.flatnonzero(
        main_bits[:candidate_limit] == reference_bits[0]
    )

    matches = []
    for candidate in candidates:
        start = int(candidate)
        if np.array_equal(
            main_bits[start:start + ref_len],
            reference_bits,
        ):
            matches.append(start)

    return matches


def search_reference_binaries(main_path, reference_paths):
    """
    Normalize main/reference inputs with the original loader and compare bits.

    This is intentionally format-independent at the logical bitstream level:
    - TXT/CSV: original loader keeps only characters 0 and 1.
    - BIN/DAT: original loader unpacks raw bytes into bits.

    If two files represent the same bit sequence after normalization, they
    match regardless of their source file format.
    """
    main_bits = load_bits(main_path)
    results = []

    for slot, path in enumerate(reference_paths, start=1):
        if not path:
            continue

        reference_bits = load_bits(path)
        offsets = find_exact_bit_occurrences(main_bits, reference_bits)

        results.append(
            ReferenceMatch(
                slot=slot,
                filename=os.path.basename(path),
                bits=reference_bits,
                offsets=offsets,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Appendix helpers
# ---------------------------------------------------------------------------

def _bit_text(bits):
    return "".join(
        "1" if int(value) else "0"
        for value in np.asarray(bits, dtype=np.uint8).ravel()
    )


def _plot_reference_signal(bits, title, filename):
    """
    Plot a reference with the same visual conventions as the original raw
    binary-signal figure.
    """
    path = os.path.join(IMG_DIR, filename)

    plt.figure(figsize=(10, 4))
    plt.plot(
        np.arange(len(bits)),
        bits,
        label="Bits (0/1)",
        linewidth=0.7,
    )
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Bit value")
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

    return path


def _reference_appendix_flowables(matches):
    """Create ReportLab flowables using the original backend report styles."""
    if not matches:
        return []

    styles = _base.build_styles()
    flowables = [
        PageBreak(),
        Paragraph(
            "Appendix — Exact Bit-Level Reference Matching",
            styles["CustomTitle"],
        ),
        Paragraph(
            "Reference files are normalized with the same input loader used "
            "for the main signal and are then compared bit by bit. Matching is "
            "therefore based on the logical binary sequence rather than on the "
            "source file byte representation. Reported positions are zero-based "
            "bit offsets and end positions are inclusive.",
            styles["CustomBody"],
        ),
        Spacer(1, 0.30 * cm),
    ]

    for result_index, match in enumerate(matches):
        ref_bits = np.asarray(match.bits, dtype=np.uint8)
        ref_len = int(ref_bits.size)

        flowables.append(
            Paragraph(
                f"Reference {match.slot}: {match.filename}",
                styles["Heading2"],
            )
        )

        summary = [
            ["Length [bits]", "Found", "Occurrences"],
            [
                f"{ref_len:,}",
                "Yes" if match.found else "No",
                f"{len(match.offsets):,}",
            ],
        ]

        summary_table = Table(
            summary,
            colWidths=[5.0 * cm, 4.0 * cm, 5.0 * cm],
            repeatRows=1,
        )
        summary_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        REPORT_THEME["primary_color"],
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        REPORT_THEME["title_font"],
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor("#B8C4D0"),
                    ),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (-1, -1),
                        REPORT_THEME["background_light"],
                    ),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ]
            )
        )
        flowables.append(summary_table)
        flowables.append(Spacer(1, 0.20 * cm))

        plot_path = _plot_reference_signal(
            ref_bits,
            (
                f"Reference {match.slot} Binary Signal "
                f"(complete input: {ref_len:,} bits)"
            ),
            f"reference_{match.slot}_signal.png",
        )

        flowables.append(
            Image(
                plot_path,
                width=REPORT_THEME["figure_width_cm"] * cm,
                height=REPORT_THEME["figure_width_cm"] * 0.5 * cm,
            )
        )
        flowables.append(
            Paragraph(
                "Complete reference bitstream using the same binary-signal "
                "representation as the main report.",
                styles["CustomCaption"],
            )
        )

        flowables.append(
            Paragraph("Bit-by-bit value", styles["Heading3"])
        )

        bit_string = _bit_text(ref_bits)
        bit_lines = [
            bit_string[index:index + 96]
            for index in range(0, len(bit_string), 96)
        ]

        # Reuse the original body style while using a fixed-width font for
        # binary values.
        from reportlab.lib.styles import ParagraphStyle

        bit_style = ParagraphStyle(
            f"ReferenceBits{match.slot}",
            parent=styles["CustomBody"],
            fontName="Courier",
            fontSize=6.6,
            leading=8.0,
        )

        flowables.append(
            Paragraph("<br/>".join(bit_lines), bit_style)
        )
        flowables.append(Spacer(1, 0.18 * cm))

        flowables.append(
            Paragraph(
                "Occurrences in main input",
                styles["Heading3"],
            )
        )

        if not match.offsets:
            flowables.append(
                Paragraph(
                    "No exact bit-level occurrence found.",
                    styles["CustomBody"],
                )
            )
        else:
            rows = [[
                "#",
                "Start bit",
                "End bit",
                "Start bit [hex]",
                "End bit [hex]",
            ]]

            for number, start in enumerate(
                match.offsets,
                start=1,
            ):
                end = start + ref_len - 1
                rows.append(
                    [
                        str(number),
                        f"{start:,}",
                        f"{end:,}",
                        f"0x{start:X}",
                        f"0x{end:X}",
                    ]
                )

            occurrence_table = Table(
                rows,
                repeatRows=1,
                colWidths=[
                    1.2 * cm,
                    3.0 * cm,
                    3.0 * cm,
                    3.8 * cm,
                    3.8 * cm,
                ],
            )
            occurrence_table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            REPORT_THEME["primary_color"],
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white,
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            REPORT_THEME["title_font"],
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.4,
                            colors.HexColor("#B8C4D0"),
                        ),
                        (
                            "ALIGN",
                            (0, 0),
                            (-1, -1),
                            "CENTER",
                        ),
                        (
                            "FONTSIZE",
                            (0, 0),
                            (-1, -1),
                            7.5,
                        ),
                    ]
                )
            )
            flowables.append(occurrence_table)

        if result_index < len(matches) - 1:
            flowables.append(PageBreak())

    return flowables


# ---------------------------------------------------------------------------
# PDF integration
# ---------------------------------------------------------------------------

def build_original_report_with_appendix(
    metrics,
    paths,
    source_file,
    matches,
):
    """
    Build the ORIGINAL report and inject the appendix into the same ReportLab
    story immediately before the original document is rendered.

    The original backend's build_pdf(), metrics, plots, CWT appearance,
    pagination logic, page header/footer and report structure are not
    reimplemented here.
    """
    _sync_paths_to_base()
    setup_dirs()

    appendix = _reference_appendix_flowables(matches)

    if not appendix:
        _base.build_pdf(
            metrics,
            paths,
            source_file=source_file,
        )
        return os.path.abspath(PDF_PATH)

    original_build = _base.SimpleDocTemplate.build

    def build_with_appendix(document, flowables, *args, **kwargs):
        # A fresh original story is created on every run, so adding the
        # appendix here is idempotent and cannot accumulate duplicate pages.
        flowables.extend(appendix)
        return original_build(
            document,
            flowables,
            *args,
            **kwargs,
        )

    _base.SimpleDocTemplate.build = build_with_appendix

    try:
        _base.build_pdf(
            metrics,
            paths,
            source_file=source_file,
        )
    finally:
        _base.SimpleDocTemplate.build = original_build

    return os.path.abspath(PDF_PATH)


# Optional compatibility alias.
def build_pdf(
    metrics,
    paths,
    source_file,
    reference_files=None,
):
    matches = (
        search_reference_binaries(
            source_file,
            reference_files or [],
        )
        if reference_files
        else []
    )

    return build_original_report_with_appendix(
        metrics,
        paths,
        source_file,
        matches,
    )
