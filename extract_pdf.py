#!/usr/bin/env python3
"""提取 PDF 论文的前 15 页内容用于解读"""
import fitz
import sys
import os

def extract_pdf_text(pdf_path, max_pages=15):
    """提取 PDF 前 max_pages 页的文本"""
    doc = fitz.open(pdf_path)
    text_parts = []

    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text = page.get_text()
        text_parts.append(f"=== Page {i+1} ===\n{text}")

    doc.close()
    return "\n\n".join(text_parts)

if __name__ == "__main__":
    pdf_path = sys.argv[1]
    if os.path.exists(pdf_path):
        print(extract_pdf_text(pdf_path))
    else:
        print(f"File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
