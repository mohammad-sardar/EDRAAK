from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from compliance_checklist import CHECKLIST_ITEMS


TOKEN_PATTERN = re.compile(r"[\u0600-\u06FFA-Za-z0-9]{2,}")


@dataclass
class ContractClause:
    clause_id: str
    title: str
    text: str


@dataclass
class LawArticleChunk:
    article_number: str
    article_label: str
    text: str


class ClausePromptBuilder:
    def build(
        self,
        clause: ContractClause,
        retrieved_articles: list[LawArticleChunk],
        checklist_labels: list[str],
    ) -> str:
        articles_block = "\n\n".join(
            f"{item.article_label}\n{item.text}" for item in retrieved_articles
        ) or "No article retrieved."
        checklist_block = ", ".join(checklist_labels) if checklist_labels else "General Saudi labor law review"
        return (
            "You are a Saudi labor law contract analyst.\n"
            "Evaluate exactly ONE employment contract clause against ONLY the retrieved Saudi Labor Law articles.\n"
            "Do not evaluate the whole contract. Do not invent legal references outside the retrieved articles.\n"
            "If retrieval is weak or insufficient, mark the clause as Needs Review and request a vector DB re-scan.\n\n"
            f"Contract Clause ID: {clause.clause_id}\n"
            f"Contract Clause Title: {clause.title}\n"
            f"Checklist Scope: {checklist_block}\n"
            f"Contract Clause Text:\n{clause.text}\n\n"
            "Retrieved Saudi Labor Law Articles:\n"
            f"{articles_block}\n\n"
            "Return strict JSON with this schema:\n"
            "{\n"
            '  "clause_id": "string",\n'
            '  "status": "Safe|Violation|Needs Review",\n'
            '  "risk_level": "High|Medium|Low",\n'
            '  "retrieval_alert": "string",\n'
            '  "violations": [\n'
            "    {\n"
            '      "article": "string",\n'
            '      "checklist_area": "string",\n'
            '      "reason": "short explanation",\n'
            '      "legal_explanation": "plain-language explanation",\n'
            '      "suggested_revision": "replacement wording or correction",\n'
            '      "comparison": {\n'
            '        "contract_text": "offending text",\n'
            '        "law_requirement": "short legal requirement"\n'
            "      }\n"
            "    }\n"
            "  ],\n"
            '  "overall_recommendation": "short recommendation"\n'
            "}\n"
        )


class SimpleVectorIndex:
    def __init__(self, documents: list[LawArticleChunk]) -> None:
        self.documents = documents
        self.doc_tokens = [self._tokenize(doc.text) for doc in documents]
        self.idf = self._build_idf(self.doc_tokens)
        self.doc_vectors = [self._vectorize(tokens) for tokens in self.doc_tokens]

    def search(self, query: str, top_k: int = 3) -> list[tuple[LawArticleChunk, float]]:
        query_vector = self._vectorize(self._tokenize(query))
        if not query_vector:
            return []
        scored: list[tuple[LawArticleChunk, float]] = []
        for document, doc_vector in zip(self.documents, self.doc_vectors):
            score = self._cosine_similarity(query_vector, doc_vector)
            if score > 0:
                scored.append((document, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _build_idf(self, tokenized_docs: list[list[str]]) -> dict[str, float]:
        total_docs = max(len(tokenized_docs), 1)
        doc_frequency: Counter[str] = Counter()
        for tokens in tokenized_docs:
            doc_frequency.update(set(tokens))
        return {
            term: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for term, freq in doc_frequency.items()
        }

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        vector = {term: count * self.idf.get(term, 0.0) for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {term: value / norm for term, value in vector.items()}

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    def _cosine_similarity(self, left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        shared_terms = set(left) & set(right)
        return sum(left[term] * right[term] for term in shared_terms)


class SaudiLaborLawIndex:
    def __init__(self, law_text: str) -> None:
        self.articles = self._split_articles(law_text)
        self.index = SimpleVectorIndex(self.articles) if self.articles else None

    def retrieve(
        self,
        clause_text: str,
        top_k: int = 3,
        article_hints: list[str] | None = None,
    ) -> list[dict[str, str | float]]:
        if not self.index:
            return []
        results = self.index.search(clause_text, top_k=max(top_k, 3))
        payload = [
            {
                "article_number": article.article_number,
                "article_label": article.article_label,
                "text": article.text,
                "score": round(score, 4),
            }
            for article, score in results
        ]
        if article_hints:
            seen = {str(item["article_number"]) for item in payload}
            for hint in article_hints:
                normalized_hint = self._normalize_digits(hint)
                if normalized_hint in seen:
                    continue
                direct_match = next(
                    (item for item in self.articles if item.article_number == normalized_hint),
                    None,
                )
                if direct_match:
                    payload.append(
                        {
                            "article_number": direct_match.article_number,
                            "article_label": direct_match.article_label,
                            "text": direct_match.text,
                            "score": 0.0,
                        }
                    )
        payload.sort(key=lambda item: float(item["score"]), reverse=True)
        unique: dict[str, dict[str, str | float]] = {}
        for item in payload:
            unique.setdefault(str(item["article_number"]), item)
        return list(unique.values())[:top_k]

    def _split_articles(self, law_text: str) -> list[LawArticleChunk]:
        normalized = re.sub(r"\s+", " ", law_text).strip()
        articles: list[LawArticleChunk] = []
        forward_pattern = re.compile(
            r"(المادة\s*\(?\s*([\d٠-٩]+)\s*\)?.*?)(?=المادة\s*\(?\s*[\d٠-٩]+\s*\)?|$)"
        )
        reverse_pattern = re.compile(
            r"(([\d٠-٩]+)\s*\)?\s*المادة.*?)(?=(?:[\d٠-٩]+\s*\)?\s*المادة|المادة\s*\(?\s*[\d٠-٩]+)|$)"
        )
        for pattern in (forward_pattern, reverse_pattern):
            for match in pattern.finditer(normalized):
                block = match.group(1).strip()
                article_number = self._normalize_digits(match.group(2))
                articles.append(
                    LawArticleChunk(
                        article_number=article_number,
                        article_label=f"المادة {article_number}",
                        text=block,
                    )
                )
        if articles:
            unique: dict[str, LawArticleChunk] = {}
            for item in articles:
                unique.setdefault(item.article_number, item)
            return list(unique.values())
        return self._build_fallback_chunks()

    def _build_fallback_chunks(self) -> list[LawArticleChunk]:
        chunks: list[LawArticleChunk] = []
        for item in CHECKLIST_ITEMS:
            for article_number in item["article_numbers"]:
                normalized_number = self._normalize_digits(article_number)
                chunks.append(
                    LawArticleChunk(
                        article_number=normalized_number,
                        article_label=f"المادة {normalized_number}",
                        text=(
                            f"المادة {normalized_number} - {item['label']}. "
                            f"{item['legal_requirement']} "
                            f"مؤشر الامتثال: {item['suggested_amendment']}"
                        ),
                    )
                )
        unique: dict[str, LawArticleChunk] = {}
        for item in chunks:
            unique.setdefault(item.article_number, item)
        return list(unique.values())

    def _normalize_digits(self, value: str) -> str:
        translation = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        return value.translate(translation)
