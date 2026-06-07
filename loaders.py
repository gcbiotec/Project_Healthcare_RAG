from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from docx import Document


def read_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def read_docx(file_path: Path) -> str:
    doc = Document(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_documents(data_dir: str) -> List[Dict]:
    docs = []
    base = Path(data_dir)

    for file_path in base.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        text = ""

        try:
            if suffix == ".txt":
                text = read_txt(file_path)
            elif suffix == ".pdf":
                text = read_pdf(file_path)
            elif suffix == ".docx":
                text = read_docx(file_path)
            else:
                continue

            if text.strip():
                docs.append({
                    "source": str(file_path),
                    "text": text.strip()
                })
        except Exception as exc:
            print(f"Erro ao ler {file_path}: {exc}")

    return docs