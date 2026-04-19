from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from analyzer import SaudiLaborContractAnalyzer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LAW_PDF = Path("C:/Users/pvp_pc/Downloads/labor-law.pdf")

app = FastAPI(title="Saudi Labor Contract Analyzer")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

analyzer = SaudiLaborContractAnalyzer(
    law_pdf_path=DEFAULT_LAW_PDF if DEFAULT_LAW_PDF.exists() else None
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    keep_app_open = request.query_params.get("app") == "1"
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "page_title": "محلل عقود العمل السعودي",
            "result": None,
            "keep_app_open": keep_app_open,
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_contract(
    request: Request,
    clauses_file: UploadFile | None = File(None),
    contract_file: UploadFile = File(...),
) -> HTMLResponse:
    try:
        result = await analyzer.analyze_uploads(
            clauses_file=clauses_file,
            contract_file=contract_file,
        )
    except ValueError as exc:
        result = {
            "summary": {
                "status": "تعذر التحليل",
                "high": 0,
                "medium": 0,
                "low": 0,
                "flagged_clauses": 0,
                "total_contract_clauses": 0,
                "safe_clauses": 0,
                "needs_review_clauses": 0,
                "total_violations": 0,
            },
            "clauses_count": 0,
            "contract_clauses_count": 0,
            "findings": [],
            "all_clause_analyses": [],
            "highlighted_contract": [],
            "obligations": {"employee": []},
            "salary_info": {"label": "غير متاح"},
            "checklist_gaps": [],
            "report_text": "",
            "error": str(exc),
        }
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "page_title": "محلل عقود العمل السعودي",
            "result": result,
            "keep_app_open": True,
        },
    )
