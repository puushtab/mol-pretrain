"""Markdown -> .docx and .pdf, for the documents we hand in.

    python scripts/md2doc.py docs/report.md docs/out/report

Handles headings, paragraphs, lists, tables, quotes, code, rules and figures
written as a standalone `![caption](path)` line, with paths relative to the
directory the command runs from.

This needs python-docx, fpdf2 and pillow, which are deliberately not project
dependencies: nothing in the pipeline imports them, and the project environment
is already slow to build on the cluster. Make a throwaway environment instead:

    uv venv ~/doctools && uv pip install --python ~/doctools/bin/python \\
        python-docx fpdf2 pillow
    ~/doctools/bin/python scripts/md2doc.py docs/report.md docs/out/report
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from fpdf import FPDF

LATIN = {
    "—": "-",
    "–": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "→": "->",
    " ": " ",
    "−": "-",
    "≥": ">=",
    "≤": "<=",
    "×": "x",
    "′": "'",
    "•": "-",
    "σ": "sd",
    "χ": "chi",
    "²": "2",
}


def latin(text: str) -> str:
    """fpdf2's core fonts are latin-1, so transliterate what they cannot encode."""
    for a, b in LATIN.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def parse(md: str) -> list[dict]:
    blocks: list[dict] = []
    lines = md.split("\n")
    i = 0
    paragraph: list[str] = []

    def flush() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append({"kind": "p", "text": " ".join(paragraph).strip()})
            paragraph = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush()
            i += 1
            continue

        if stripped.startswith("```"):
            flush()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            blocks.append({"kind": "code", "text": "\n".join(code)})
            continue

        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            flush()
            blocks.append({"kind": "rule"})
            i += 1
            continue

        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image:
            flush()
            blocks.append({"kind": "image", "caption": image.group(1), "path": image.group(2)})
            i += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            flush()
            blocks.append({"kind": "h", "level": len(heading.group(1)), "text": heading.group(2)})
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                blocks.append({"kind": "table", "rows": rows})
            continue

        if stripped.startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append({"kind": "quote", "text": " ".join(q for q in quote if q)})
            continue

        bullet = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        number = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if bullet or number:
            flush()
            items = []
            while i < len(lines):
                b = re.match(r"^(\s*)[-*]\s+(.*)$", lines[i])
                n = re.match(r"^(\s*)(\d+)\.\s+(.*)$", lines[i])
                if b:
                    items.append(
                        {"indent": len(b.group(1)) // 2, "text": b.group(2), "ordered": False}
                    )
                elif n:
                    items.append(
                        {"indent": len(n.group(1)) // 2, "text": n.group(3), "ordered": True}
                    )
                elif lines[i].strip() and lines[i].startswith(("   ", "\t")) and items:
                    items[-1]["text"] += " " + lines[i].strip()
                else:
                    break
                i += 1
            blocks.append({"kind": "list", "items": items})
            continue

        paragraph.append(stripped)
        i += 1

    flush()
    return blocks


def spans(text: str) -> list[tuple[str, set[str]]]:
    """Split inline markdown into (text, styles) pairs."""
    out: list[tuple[str, set[str]]] = []
    pattern = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*|\[[^\]]+\]\([^)]+\))")
    for piece in pattern.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            out.append((piece[2:-2], {"bold"}))
        elif piece.startswith("`") and piece.endswith("`"):
            out.append((piece[1:-1], {"code"}))
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            out.append((piece[1:-1], {"italic"}))
        elif piece.startswith("["):
            label = re.match(r"\[([^\]]+)\]", piece)
            out.append((label.group(1) if label else piece, set()))
        else:
            out.append((piece, set()))
    return out


def write_docx(blocks: list[dict], path: Path) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    def runs(paragraph, text, base=None):
        for piece, marks in spans(text):
            run = paragraph.add_run(piece)
            run.bold = "bold" in marks or (base == "bold")
            run.italic = "italic" in marks
            if "code" in marks:
                run.font.name = "Consolas"
                run.font.size = Pt(9.5)

    for block in blocks:
        kind = block["kind"]
        if kind == "h":
            doc.add_heading(re.sub(r"[*`]", "", block["text"]), level=min(block["level"], 4))
        elif kind == "p":
            runs(doc.add_paragraph(), block["text"])
        elif kind == "quote":
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Pt(24)
            runs(paragraph, block["text"])
            for run in paragraph.runs:
                run.italic = True
                run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
        elif kind == "code":
            paragraph = doc.add_paragraph()
            run = paragraph.add_run(block["text"])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif kind == "image":
            source = Path(block["path"])
            if source.exists():
                doc.add_picture(str(source), width=Inches(6.2))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                if block["caption"]:
                    caption = doc.add_paragraph(block["caption"])
                    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in caption.runs:
                        run.italic = True
                        run.font.size = Pt(9)
        elif kind == "rule":
            doc.add_paragraph("_" * 60).alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif kind == "list":
            for item in block["items"]:
                name = "List Number" if item["ordered"] else "List Bullet"
                suffix = "" if item["indent"] == 0 else f" {min(item['indent'] + 1, 3)}"
                try:
                    paragraph = doc.add_paragraph(style=name + suffix)
                except KeyError:
                    paragraph = doc.add_paragraph(style=name)
                runs(paragraph, item["text"])
        elif kind == "table":
            rows = block["rows"]
            table = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
            table.style = "Light Grid Accent 1"
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    target = table.cell(r, c).paragraphs[0]
                    runs(target, cell, base="bold" if r == 0 else None)
                    for run in target.runs:
                        run.font.size = Pt(9)
    doc.save(path)


class PDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"{self.page_no()}", align="C")


def write_pdf(blocks: list[dict], path: Path) -> None:
    pdf = PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    def emit(text: str, size: float, style: str = "", gap: float = 4.2, indent: float = 0.0):
        pdf.set_font("Helvetica", style, size)
        pdf.set_x(pdf.l_margin + indent)
        for piece, marks in spans(text):
            flags = ("B" if "bold" in marks or "B" in style else "") + (
                "I" if "italic" in marks or "I" in style else ""
            )
            pdf.set_font(
                "Courier" if "code" in marks else "Helvetica",
                flags,
                size - (0.5 if "code" in marks else 0),
            )
            pdf.write(gap + 1.2, latin(piece))
        pdf.ln(gap + 1.2)

    for block in blocks:
        kind = block["kind"]
        if kind == "h":
            level = block["level"]
            pdf.ln(3 if level > 1 else 5)
            pdf.set_text_color(0, 0, 0)
            emit(
                re.sub(r"[*`]", "", block["text"]),
                {1: 17, 2: 13, 3: 11.2, 4: 10.4}[min(level, 4)],
                "B",
                5,
            )
            if level <= 2:
                pdf.set_draw_color(200, 200, 200)
                pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
                pdf.ln(2.5)
        elif kind == "p":
            pdf.set_text_color(25, 25, 25)
            emit(block["text"], 10)
            pdf.ln(1.6)
        elif kind == "quote":
            pdf.set_text_color(90, 90, 90)
            emit(block["text"], 9.6, "I", indent=6)
            pdf.ln(1.6)
        elif kind == "code":
            pdf.set_font("Courier", "", 8.4)
            pdf.set_text_color(40, 40, 40)
            for line in block["text"].split("\n"):
                pdf.set_x(pdf.l_margin + 4)
                pdf.cell(0, 4.2, latin(line), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif kind == "image":
            source = Path(block["path"])
            if source.exists():
                from PIL import Image

                with Image.open(source) as handle:
                    ratio = handle.height / handle.width
                draw = min(width, 160)
                if pdf.get_y() + draw * ratio + 10 > pdf.h - 18:
                    pdf.add_page()
                pdf.image(str(source), x=pdf.l_margin + (width - draw) / 2, w=draw)
                if block["caption"]:
                    pdf.ln(1)
                    pdf.set_font("Helvetica", "I", 8.5)
                    pdf.set_text_color(110, 110, 110)
                    pdf.multi_cell(0, 4, latin(block["caption"]), align="C")
                pdf.ln(3)
        elif kind == "rule":
            pdf.ln(2)
            pdf.set_draw_color(215, 215, 215)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)
        elif kind == "list":
            pdf.set_text_color(25, 25, 25)
            for n, item in enumerate(block["items"], start=1):
                marker = f"{n}. " if item["ordered"] else "- "
                emit(marker + item["text"], 10, indent=5 + 5 * item["indent"], gap=3.6)
            pdf.ln(1.8)
        elif kind == "table":
            rows = block["rows"]
            columns = max(len(r) for r in rows)
            col_width = width / columns
            pdf.ln(1)
            for r, row in enumerate(rows):
                cells = [latin(re.sub(r"[*`]", "", c)) for c in row] + [""] * (columns - len(row))
                pdf.set_font("Helvetica", "B" if r == 0 else "", 8.2)
                height = 4.4 * max(
                    1,
                    max(
                        len(pdf.multi_cell(col_width, 4.4, c, dry_run=True, output="LINES"))
                        for c in cells
                    ),
                )
                if pdf.get_y() + height > pdf.h - 18:
                    pdf.add_page()
                y = pdf.get_y()
                for c, cell in enumerate(cells):
                    pdf.set_xy(pdf.l_margin + c * col_width, y)
                    pdf.set_fill_color(238, 238, 238)
                    pdf.multi_cell(
                        col_width,
                        4.4,
                        cell,
                        border=1,
                        align="L",
                        fill=(r == 0),
                        max_line_height=4.4,
                    )
                pdf.set_y(y + height)
            pdf.ln(3)

    pdf.output(str(path))


def main() -> None:
    source = Path(sys.argv[1])
    base = Path(sys.argv[2])
    base.parent.mkdir(parents=True, exist_ok=True)
    blocks = parse(source.read_text())
    write_docx(blocks, base.with_suffix(".docx"))
    write_pdf(blocks, base.with_suffix(".pdf"))
    print(f"{base.with_suffix('.docx')}\n{base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
