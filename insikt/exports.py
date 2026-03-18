from __future__ import annotations

import os
import tempfile

def export_text(text):
    return text.encode("utf-8")


def export_docx(text):
    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_paragraph(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        with open(tmp.name, "rb") as handle:
            data = handle.read()
    os.unlink(tmp.name)
    return data


def export_pdf(text):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as handle:
            data = handle.read()
    os.unlink(tmp.name)
    return data


def export_markdown(text):
    return text.encode("utf-8")
