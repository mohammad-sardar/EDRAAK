from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader

from rag_pipeline import ContractClause

logging.getLogger("pypdf").setLevel(logging.ERROR)


CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"(?:^|\n|\s)("
    r"(?:البند|المادة)\s+(?:\d+|[٠-٩]+|الأول|الاول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)\s*[:\-]?"
    r"|(?:أولا|أولاً|ثانيا|ثانياً|ثالثا|ثالثاً|رابعا|رابعاً|خامسا|خامساً|سادسا|سادساً)\s*[:\-]"
    r"|\d+\s*[-.)]"
    r")",
    flags=re.IGNORECASE,
)


class DocumentParser:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

    def parse_bytes(self, filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")
        if suffix == ".pdf":
            return self._parse_pdf(content)
        if suffix == ".docx":
            return self._parse_docx(content)
        return content.decode("utf-8", errors="ignore")

    def parse_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if suffix == ".docx":
            return self._parse_docx(path.read_bytes())
        return path.read_text(encoding="utf-8", errors="ignore")

    def split_into_clauses(self, text: str) -> list[str]:
        normalized = self.normalize_text(text)
        normalized = self._inject_clause_breaks(normalized)
        boundaries = [match.start(1) for match in CLAUSE_BOUNDARY_PATTERN.finditer(normalized)]
        clauses: list[str] = []
        if boundaries:
            for index, start in enumerate(boundaries):
                end = boundaries[index + 1] if index + 1 < len(boundaries) else len(normalized)
                clause = normalized[start:end].strip(" -\n\r\t")
                if len(clause) > 20:
                    clauses.append(clause)
        if clauses:
            return clauses
        fallback_parts = re.split(r"(?<=[.!؟])\s+|\n{2,}", normalized)
        return [part.strip() for part in fallback_parts if len(part.strip()) > 20]

    def extract_contract_clauses(self, text: str) -> list[ContractClause]:
        raw_clauses = self.split_into_clauses(text)
        return [
            ContractClause(
                clause_id=f"clause-{index}",
                title=self._infer_clause_title(clause, index),
                text=clause,
            )
            for index, clause in enumerate(raw_clauses, start=1)
        ]

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = text.replace("\ufeff", "")
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _inject_clause_breaks(self, text: str) -> str:
        patterns = [
            r"\s+(?=(?:البند|المادة)\s+(?:\d+|[٠-٩]+|الأول|الاول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر))",
            r"\s+(?=(?:أولا|أولاً|ثانيا|ثانياً|ثالثا|ثالثاً|رابعا|رابعاً|خامسا|خامساً|سادسا|سادساً)\s*[:\-])",
            r"\s+(?=\d+\s*[-.)])",
        ]
        updated = text
        for pattern in patterns:
            updated = re.sub(pattern, "\n", updated)
        return updated

    def _parse_pdf(self, content: bytes) -> str:
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    def _parse_docx(self, content: bytes) -> str:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml_data = archive.read("word/document.xml").decode("utf-8", errors="ignore")
        text = re.sub(r"</w:p>", "\n", xml_data)
        text = re.sub(r"<[^>]+>", "", text)
        return self.normalize_text(text)

    def _infer_clause_title(self, clause: str, index: int) -> str:
        first_line = clause.splitlines()[0].strip()
        cleaned = re.sub(r"^(?:البند|المادة)\s+(?:\d+|[٠-٩]+|الأول|الاول|الثاني|الثالث|الرابع|الخامس|السادس|السابع|الثامن|التاسع|العاشر)\s*[:\-]?\s*", "", first_line)
        cleaned = re.sub(r"^(?:أولا|أولاً|ثانيا|ثانياً|ثالثا|ثالثاً|رابعا|رابعاً|خامسا|خامساً|سادسا|سادساً)\s*[:\-]?\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:")
        if not cleaned:
            return f"البند {index}"
        if len(cleaned) > 55:
            cleaned = cleaned[:55].rsplit(" ", 1)[0].strip()
        return cleaned
