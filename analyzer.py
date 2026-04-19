from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from compliance_checklist import CHECKLIST_ITEMS, SEVERITY_ORDER
from document_parser import DocumentParser
from rag_pipeline import ClausePromptBuilder, ContractClause, LawArticleChunk, SaudiLaborLawIndex


DATE_PATTERN = re.compile(r"(?:اليوم|يوم)\s*(\d{1,2})|(?:أول|اول)\s+كل\s+شهر|(?:نهاية|اخر|آخر)\s+كل\s+شهر")
OVERTIME_RATE_PATTERN = re.compile(r"(?:150\s*%|50\s*%|مرة\s+ونصف|نصف\s+أجره\s+الإضافي)", re.IGNORECASE)
FAMILY_SCOPE_PATTERN = re.compile(r"(?:العائلة|عائلته|عائلتهم|أسرته|الأسرة|تابعيه|المعالين)")


@dataclass
class LawArticleSnippet:
    article: str
    snippet: str


class SaudiLaborContractAnalyzer:
    def __init__(self, law_pdf_path: Path | None = None) -> None:
        self.parser = DocumentParser()
        self.law_pdf_path = law_pdf_path
        self.law_text = self._load_law_text(law_pdf_path) if law_pdf_path else ""
        self.law_index = SaudiLaborLawIndex(self.law_text) if self.law_text else None
        self.prompt_builder = ClausePromptBuilder()

    async def analyze_uploads(
        self,
        clauses_file: UploadFile | None,
        contract_file: UploadFile,
    ) -> dict[str, Any]:
        contract_text = self.parser.parse_bytes(
            contract_file.filename or "contract.txt",
            await contract_file.read(),
        )
        supplemental_reference_clauses = 0
        custom_clauses_used = False
        if clauses_file and clauses_file.filename:
            clauses_bytes = await clauses_file.read()
            if clauses_bytes:
                clauses_text = self.parser.parse_bytes(clauses_file.filename, clauses_bytes)
                supplemental_reference_clauses = len(self.parser.split_into_clauses(clauses_text))
                custom_clauses_used = supplemental_reference_clauses > 0

        contract_clauses = self.parser.extract_contract_clauses(contract_text)
        analyzed_clauses = [self._analyze_clause(clause) for clause in contract_clauses]
        all_clause_analyses = self._sort_analyses_by_risk(analyzed_clauses)
        findings = [item for item in all_clause_analyses if item["status"] == "Violation"]
        summary = self._summarize_findings(analyzed_clauses)
        highlighted_contract = self._build_highlighted_contract(all_clause_analyses)
        salary_info = self._extract_salary_info(contract_text)
        obligations = self._extract_obligations(all_clause_analyses)
        revised_contract = self._build_revised_contract(all_clause_analyses)
        checklist_gaps = self._build_checklist_gaps(all_clause_analyses)
        report_text = self._build_report_text(
            summary=summary,
            analyses=all_clause_analyses,
            salary_info=salary_info,
            obligations=obligations,
            checklist_gaps=checklist_gaps,
            custom_clauses_used=custom_clauses_used,
        )
        return {
            "summary": summary,
            "clauses_count": supplemental_reference_clauses,
            "contract_clauses_count": len(contract_clauses),
            "findings": findings,
            "all_clause_analyses": all_clause_analyses,
            "clauses": all_clause_analyses,
            "json_output": all_clause_analyses,
            "highlighted_contract": highlighted_contract,
            "custom_clauses_used": custom_clauses_used,
            "salary_info": salary_info,
            "obligations": obligations,
            "revised_contract": revised_contract,
            "checklist_gaps": checklist_gaps,
            "report_text": report_text,
            "rag_pipeline": {
                "mode": "strict_clause_by_clause_checklist",
                "retrieval_top_k": 3,
                "clause_processing_count": len(all_clause_analyses),
                "law_articles_indexed": len(self.law_index.articles) if self.law_index else 0,
                "aggregation_mode": "append_all_clause_results",
                "strict_checklist_areas": [item["label"] for item in CHECKLIST_ITEMS],
            },
        }

    def _analyze_clause(self, clause: ContractClause) -> dict[str, Any]:
        matched_checkpoints = self._identify_relevant_checkpoints(clause.text)
        article_hints = [article for item in matched_checkpoints for article in item["article_numbers"]]
        retrieved_articles = self._retrieve_law_articles(clause.text, top_k=3, article_hints=article_hints)
        retrieval_alert = self._build_retrieval_alert(article_hints, retrieved_articles)
        prompt_articles = [
            LawArticleChunk(
                article_number=str(item["article_number"]),
                article_label=str(item["article_label"]),
                text=str(item["text"]),
            )
            for item in retrieved_articles
        ]
        evaluation_prompt = self.prompt_builder.build(
            clause,
            prompt_articles,
            [item["label"] for item in matched_checkpoints],
        )

        violations: list[dict[str, Any]] = []
        for checkpoint in matched_checkpoints:
            issue = self._evaluate_checkpoint(clause.text, checkpoint, retrieved_articles)
            if issue:
                violations.append(issue)

        if violations:
            status = "Violation"
            risk_level = max(violations, key=lambda item: SEVERITY_ORDER[item["risk"]])["risk"]
            risk_score = min(sum(item["weight"] for item in violations) * 8, 100)
        elif retrieval_alert:
            status = "Needs Review"
            risk_level = "Medium"
            risk_score = 35
        else:
            status = "Safe"
            risk_level = "Low"
            risk_score = 0

        combined_articles = sorted({issue["article"] for issue in violations})
        legal_explanation = " ".join(issue["legal_explanation"] for issue in violations[:2]) if violations else "لم تظهر مخالفة واضحة في هذا البند ضمن checklist الحالية."
        suggested_amendment = violations[0]["suggestion"] if violations else "لا يتطلب هذا البند تعديلا ظاهرا في الفحص الحالي."

        return {
            "clause_id": clause.clause_id,
            "clause_title": clause.title,
            "clause_text": clause.text,
            "clause_label": self._build_clause_label(clause.text),
            "status": status,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "matched_checkpoints": [item["label"] for item in matched_checkpoints],
            "retrieval_alert": retrieval_alert,
            "retrieved_articles": retrieved_articles,
            "evaluation_prompt": evaluation_prompt,
            "violations": violations,
            "issues": violations,
            "article_list": combined_articles,
            "legal_explanation": legal_explanation,
            "suggested_amendment": suggested_amendment,
            "recommended_text": suggested_amendment,
        }

    def _identify_relevant_checkpoints(self, clause_text: str) -> list[dict[str, Any]]:
        normalized = clause_text.lower()
        matched: list[dict[str, Any]] = []
        for item in CHECKLIST_ITEMS:
            if any(keyword.lower() in normalized for keyword in item["keywords"]):
                matched.append(item)
                continue
            for check in item["checks"]:
                pattern = check.get("pattern")
                if pattern and re.search(pattern, clause_text, flags=re.IGNORECASE):
                    matched.append(item)
                    break
        return matched

    def _evaluate_checkpoint(
        self,
        clause_text: str,
        checkpoint: dict[str, Any],
        retrieved_articles: list[dict[str, str | float]],
    ) -> dict[str, Any] | None:
        for check in checkpoint["checks"]:
            outcome = self._run_check(clause_text, checkpoint, check)
            if not outcome:
                continue
            article_label = self._format_article_label(checkpoint["article_numbers"])
            snippet = self._select_law_snippet(checkpoint["article_numbers"], retrieved_articles)
            return {
                "title": checkpoint["label"],
                "checklist_area": checkpoint["label"],
                "risk": checkpoint["severity"],
                "article": article_label,
                "message": check["message"],
                "legal_explanation": checkpoint["legal_requirement"],
                "suggestion": checkpoint["suggested_amendment"],
                "law_snippet": snippet,
                "weight": checkpoint["weight"],
                "comparison": {
                    "contract_text": self._build_clause_label(clause_text),
                    "law_requirement": checkpoint["legal_requirement"],
                },
            }
        return None

    def _run_check(self, clause_text: str, checkpoint: dict[str, Any], check: dict[str, Any]) -> bool:
        check_type = check["type"]
        pattern = check.get("pattern")
        if check_type == "numeric_max":
            match = re.search(pattern, clause_text, flags=re.IGNORECASE) if pattern else None
            return bool(match and int(match.group(1)) > check["limit"])
        if check_type == "numeric_min":
            match = re.search(pattern, clause_text, flags=re.IGNORECASE) if pattern else None
            return bool(match and int(match.group(1)) < check["limit"])
        if check_type == "contains_pattern":
            return bool(pattern and re.search(pattern, clause_text, flags=re.IGNORECASE))
        if check_type == "missing_specific_date":
            lower = clause_text.lower()
            salary_context = any(keyword.lower() in lower for keyword in checkpoint["keywords"])
            return salary_context and not DATE_PATTERN.search(clause_text)
        if check_type == "missing_overtime_rate":
            lower = clause_text.lower()
            overtime_context = any(keyword.lower() in lower for keyword in checkpoint["keywords"])
            return overtime_context and not OVERTIME_RATE_PATTERN.search(clause_text)
        if check_type == "missing_family_scope":
            lower = clause_text.lower()
            insurance_context = any(keyword.lower() in lower for keyword in checkpoint["keywords"])
            return insurance_context and not FAMILY_SCOPE_PATTERN.search(clause_text)
        return False

    def _build_retrieval_alert(
        self,
        article_hints: list[str],
        retrieved_articles: list[dict[str, str | float]],
    ) -> str | None:
        if not article_hints and retrieved_articles:
            return None
        if not retrieved_articles:
            return "لم يتم العثور على مواد قانونية مرتبطة بهذا البند. يُنصح بإعادة فحص قاعدة المواد القانونية لهذا البند."
        retrieved_numbers = {str(item["article_number"]) for item in retrieved_articles}
        missing = [article for article in article_hints if article not in retrieved_numbers]
        if missing:
            return f"لم يتم استرجاع بعض المواد المتوقعة لهذا البند ({', '.join(missing)}). يُنصح بإعادة فحص قاعدة المواد القانونية لهذا البند."
        return None

    def _summarize_findings(self, analyses: list[dict[str, Any]]) -> dict[str, Any]:
        high = medium = low = total_violations = 0
        for analysis in analyses:
            for violation in analysis["violations"]:
                total_violations += 1
                if violation["risk"] == "High":
                    high += 1
                elif violation["risk"] == "Medium":
                    medium += 1
                else:
                    low += 1
        status = "Needs Legal Review" if high else "Moderate Review" if medium else "Preliminarily Safe"
        return {
            "status": status,
            "high": high,
            "medium": medium,
            "low": low,
            "total_contract_clauses": len(analyses),
            "flagged_clauses": sum(1 for item in analyses if item["status"] == "Violation"),
            "safe_clauses": sum(1 for item in analyses if item["status"] == "Safe"),
            "needs_review_clauses": sum(1 for item in analyses if item["status"] == "Needs Review"),
            "total_violations": total_violations,
        }

    def _build_highlighted_contract(self, analyses: list[dict[str, Any]]) -> list[dict[str, str]]:
        highlighted: list[dict[str, str]] = []
        for index, analysis in enumerate(analyses, start=1):
            if analysis["status"] == "Violation" and analysis["violations"]:
                primary = analysis["violations"][0]
                text = f"البند {index}: {analysis['clause_label']} - مخالفة وفق {primary['article']}."
                severity = "danger" if analysis["risk_level"] == "High" else "warning"
            elif analysis["status"] == "Needs Review":
                text = f"البند {index}: {analysis['clause_label']} - يحتاج إعادة فحص قانوني."
                severity = "warning"
            else:
                text = f"البند {index}: {analysis['clause_label']} - سليم مبدئيا."
                severity = "safe"
            highlighted.append({"text": text, "severity": severity})
        return highlighted

    def _sort_analyses_by_risk(self, analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        status_order = {"Violation": 0, "Needs Review": 1, "Safe": 2}
        return sorted(
            analyses,
            key=lambda item: (
                status_order.get(item["status"], 3),
                -SEVERITY_ORDER.get(item["risk_level"], 0),
                -int(item.get("risk_score", 0)),
            ),
        )

    def _retrieve_law_articles(
        self,
        clause_text: str,
        top_k: int = 3,
        article_hints: list[str] | None = None,
    ) -> list[dict[str, str | float]]:
        if not self.law_index:
            return []
        return self.law_index.retrieve(clause_text, top_k=top_k, article_hints=article_hints)

    def _build_clause_label(self, clause: str) -> str:
        compact = self.parser.normalize_text(clause).replace("\n", " ")
        compact = re.sub(r"\s+", " ", compact).strip()
        compact = re.sub(r"^(?:البند|المادة)\s+\d+\s*[:\-]?\s*", "", compact)
        if len(compact) <= 80:
            return compact
        short = compact[:80].rsplit(" ", 1)[0].strip()
        return f"{short}..."

    def _extract_salary_info(self, contract_text: str) -> dict[str, str | None]:
        patterns = [
            r"(?:الراتب(?:\s+الأساسي)?|الأجر(?:\s+الأساسي)?)[^\n]{0,30}?(\d[\d,\.]*)\s*(?:ريال|ر\.س|SAR)",
            r"(\d[\d,\.]*)\s*(?:ريال|ر\.س|SAR)[^\n]{0,20}?(?:شهري|شهريا|شهريًا)",
        ]
        for pattern in patterns:
            match = re.search(pattern, contract_text, flags=re.IGNORECASE)
            if match:
                amount = match.group(1).replace(",", "").strip()
                return {
                    "amount": amount,
                    "currency": "ريال سعودي",
                    "label": f"{amount} ريال سعودي",
                }
        return {"amount": None, "currency": None, "label": "لم يتم رصد راتب واضح في العقد"}

    def _extract_obligations(self, analyses: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
        employee: list[dict[str, str]] = []
        for analysis in analyses:
            clause_text = self.parser.normalize_text(analysis["clause_text"])
            if any(marker in clause_text for marker in ("يلتزم الموظف", "يلتزم العامل", "يجب على الموظف", "يجب على العامل")):
                employee.append({"text": self._extract_obligation_phrase(clause_text)})
        if not employee:
            employee = [{"text": "الالتزام بساعات العمل والمهام المحددة في العقد بما لا يخالف النظام."}]
        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in employee:
            if item["text"] in seen:
                continue
            seen.add(item["text"])
            deduped.append(item)
        return {"employee": deduped[:6]}

    def _extract_obligation_phrase(self, text: str) -> str:
        cleaned = re.sub(r"^(?:البند|المادة)\s+\d+\s*[:\-]?\s*", "", text).strip()
        parts = re.split(r"[.\n]|،", cleaned)
        for part in parts:
            part = re.sub(r"\s+", " ", part).strip(" ،-")
            if any(marker in part for marker in ("يلتزم", "يتعهد", "يجب على")):
                return part if len(part) <= 100 else f"{part[:100].rsplit(' ', 1)[0]}..."
        return cleaned if len(cleaned) <= 100 else f"{cleaned[:100].rsplit(' ', 1)[0]}..."

    def _build_report_text(
        self,
        summary: dict[str, Any],
        analyses: list[dict[str, Any]],
        salary_info: dict[str, str | None],
        obligations: dict[str, list[dict[str, str]]],
        checklist_gaps: list[str],
        custom_clauses_used: bool,
    ) -> str:
        lines = [
            "تقرير تحليل عقد العمل السعودي",
            "",
            f"الحالة العامة: {summary['status']}",
            f"إجمالي البنود: {summary['total_contract_clauses']}",
            f"البنود المخالفة: {summary['flagged_clauses']}",
            f"إجمالي المخالفات: {summary['total_violations']}",
            f"الراتب المستخرج: {salary_info['label']}",
            f"تم استخدام ملف بنود إضافي: {'نعم' if custom_clauses_used else 'لا'}",
            "",
            "التزامات الموظف:",
        ]
        lines.extend(f"- {item['text']}" for item in obligations.get("employee", []))
        if checklist_gaps:
            lines.append("")
            lines.append("محاور تحتاج مراجعة إضافية على مستوى العقد:")
            lines.extend(f"- {item}" for item in checklist_gaps)
        lines.append("")
        lines.append("نتائج البنود:")
        for analysis in analyses:
            lines.append(f"- {analysis['clause_title']}: {analysis['status']}")
            if analysis["violations"]:
                for violation in analysis["violations"]:
                    lines.append(f"  المادة: {violation['article']}")
                    lines.append(f"  السبب: {violation['message']}")
                    lines.append(f"  التعديل المقترح: {violation['suggestion']}")
            elif analysis["retrieval_alert"]:
                lines.append(f"  تنبيه الاسترجاع: {analysis['retrieval_alert']}")
        return "\n".join(lines)

    def _build_revised_contract(self, analyses: list[dict[str, Any]]) -> list[dict[str, str]]:
        revised = []
        for index, clause in enumerate(analyses, start=1):
            severity = "danger" if clause["risk_level"] == "High" else "warning" if clause["risk_level"] == "Medium" else "safe"
            revised.append(
                {
                    "title": f"البند {index}",
                    "original": clause["clause_text"],
                    "status": clause["status"],
                    "severity": severity,
                    "suggested": clause["recommended_text"],
                }
            )
        return revised

    def _build_checklist_gaps(self, analyses: list[dict[str, Any]]) -> list[str]:
        covered = {label for analysis in analyses for label in analysis["matched_checkpoints"]}
        return [item["label"] for item in CHECKLIST_ITEMS if item["label"] not in covered]

    def _select_law_snippet(
        self,
        article_numbers: list[str],
        retrieved_articles: list[dict[str, str | float]],
    ) -> str:
        normalized_hints = {str(article) for article in article_numbers}
        for item in retrieved_articles:
            if str(item["article_number"]) in normalized_hints:
                return str(item["text"])[:280]
        if self.law_index:
            for article_number in article_numbers:
                direct_match = next(
                    (
                        item
                        for item in self.law_index.articles
                        if str(item.article_number) == str(article_number)
                    ),
                    None,
                )
                if direct_match:
                    return direct_match.text[:280]
        requested_label = self._format_article_label(article_numbers)
        return f"تعذر استخراج مقتطف مطابق لـ {requested_label}، لذلك تم إخفاء أي مقتطف غير مطابق لتفادي عرض مادة مختلفة."

    def _format_article_label(self, article_numbers: list[str]) -> str:
        if len(article_numbers) == 1:
            return f"المادة {article_numbers[0]}"
        return "المواد " + " و".join(article_numbers)

    def _load_law_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        text = self.parser.parse_file(path)
        text = self.parser.normalize_text(text)
        return re.sub(r"\s+", " ", text)
