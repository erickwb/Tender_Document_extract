from pathlib import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


def pdf_to_pages(pdf_path: Path) -> list[str]:
    pdf_path = Path(pdf_path).expanduser().resolve(strict=True)

    pages: list[str] = []

    for layout in extract_pages(pdf_path):
        page_text: list[str] = []
        for element in layout:
            if isinstance(element, LTTextContainer):
                page_text.append(element.get_text())

        pages.append("".join(page_text).strip())

    return pages
