"""
Note Compiler Module.

Accumulates OCR results across multiple key frames and compiles
them into clean, formatted lecture notes.

Strategy: Keep the BEST (most complete) extraction as the master
version. Each new extraction either:
  - Replaces the master if it has more content
  - Is discarded if it's a subset of the master
  - Starts a new section if board was erased (very different content)
"""

import difflib
import os
from fpdf import FPDF
import config


class NoteCompiler:
    """
    Accumulates and merges OCR extractions into clean lecture notes.

    Uses a "best extraction wins" strategy rather than merging fragments.
    This avoids garbled output from combining multiple partial reads.
    """

    def __init__(self):
        self._sections = []          # Completed sections (after board erase)
        self._current_best = []      # Best lines for current section
        self._current_best_len = 0   # Total character count of best
        self._extraction_count = 0

    def add_extraction(self, lines):
        """
        Add a new OCR extraction result.

        Parameters
        ----------
        lines : list of str
            Text lines from a single OCR extraction, in reading order.
        """
        if not lines:
            return

        self._extraction_count += 1
        new_text = "\n".join(lines)
        new_len = len(new_text)

        if not self._current_best:
            # First extraction — just store it
            self._current_best = lines[:]
            self._current_best_len = new_len
            return

        current_text = "\n".join(self._current_best)

        # Check similarity between current best and new extraction
        similarity = difflib.SequenceMatcher(
            None, current_text.lower(), new_text.lower()
        ).ratio()

        if similarity < 0.15:
            # Very different content — board was likely erased
            # Save current section and start new one
            if self._current_best:
                self._sections.append(self._current_best[:])
            self._current_best = lines[:]
            self._current_best_len = new_len
            return

        if similarity > 0.95:
            # Nearly identical — skip
            return

        # Partial overlap — keep the version with MORE total text content
        # This handles the case where teacher moves away and more text is visible
        if new_len > self._current_best_len:
            self._current_best = lines[:]
            self._current_best_len = new_len

    def get_notes(self):
        """
        Get the compiled lecture notes as a formatted string.

        Returns
        -------
        str
            Clean, formatted lecture notes.
        """
        # Finalize current section
        all_sections = self._sections[:]
        if self._current_best:
            all_sections.append(self._current_best[:])

        if not all_sections:
            return "No notes were extracted from the video."

        lines = []
        lines.append("=" * 56)
        lines.append("             LECTURE NOTES")
        lines.append("=" * 56)
        lines.append("")

        for i, section in enumerate(all_sections):
            if i > 0:
                lines.append("")
                lines.append("-" * 40)
                lines.append(f"  [Board Section {i + 1}]")
                lines.append("-" * 40)
                lines.append("")

            for text_line in section:
                lines.append(text_line)

        lines.append("")
        lines.append("=" * 56)
        lines.append(f"  Total sections: {len(all_sections)}")
        lines.append(f"  Total OCR extractions: {self._extraction_count}")
        lines.append("=" * 56)

        return "\n".join(lines)

    def save_txt(self, filepath=None):
        """Save notes as a text file."""
        if filepath is None:
            filepath = config.NOTES_TXT

        notes = self.get_notes()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(notes)

        print(f"[Notes] Saved text notes to: {filepath}")
        return filepath

    def save_pdf(self, filepath=None):
        """Save notes as a PDF file."""
        if filepath is None:
            filepath = config.NOTES_PDF

        notes = self.get_notes()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "Lecture Notes", ln=True, align="C")
        pdf.ln(8)

        # Content
        pdf.set_font("Helvetica", "", 11)
        effective_width = pdf.w - pdf.l_margin - pdf.r_margin

        for line in notes.split("\n"):
            line = line.strip()

            if not line:
                pdf.ln(4)
                continue

            # Handle section separators
            if line.startswith("=") or line.startswith("-"):
                pdf.set_font("Helvetica", "", 9)
                # Truncate separator to fit page width
                pdf.cell(0, 5, line[:80], ln=True)
                pdf.set_font("Helvetica", "", 11)
            elif line.startswith("[Board Section"):
                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 10, line, ln=True)
                pdf.set_font("Helvetica", "", 11)
            else:
                # Encode to latin-1, replacing unsupported chars
                safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
                try:
                    pdf.multi_cell(effective_width, 7, safe_line)
                except Exception:
                    # Fallback: use cell with truncation
                    pdf.cell(0, 7, safe_line[:100], ln=True)

        pdf.output(filepath)
        print(f"[Notes] Saved PDF notes to: {filepath}")
        return filepath

    def get_extraction_count(self):
        """Return how many OCR extractions were performed."""
        return self._extraction_count

    def get_section_count(self):
        """Return how many board sections were detected."""
        count = len(self._sections)
        if self._current_best:
            count += 1
        return count
