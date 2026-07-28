import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.reports.report_service import get_report_service
from app.schemas.report import ReportResponse

router = APIRouter()

_EXPORT_CONFIG = {
    "json": ("application/json", "json"),
    "pdf": ("application/pdf", "pdf"),
    "excel": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xlsx",
    ),
}


@router.get("/reports/{student_id}", response_model=ReportResponse)
def get_report(student_id: str):
    """
    Task 2 deliverable: total attempts, correct vs incorrect, overall
    score, gesture-wise performance, most difficult gestures, average
    confidence, average response time, and improvement across attempts.
    """
    return get_report_service().generate_report(student_id)


@router.get("/reports/{student_id}/export")
def export_report(
    student_id: str,
    format: str = Query(default="json", pattern="^(json|pdf|excel)$"),
):
    """
    Downloads the report as a file. `format` is one of json (mandatory),
    pdf or excel (bonus - require reportlab / openpyxl respectively;
    returns a clear error if the optional dependency isn't installed
    rather than a raw exception).
    """
    content_type, extension = _EXPORT_CONFIG[format]
    output_path = Path(tempfile.mkdtemp()) / f"assessment_report_{student_id}.{extension}"

    report_service = get_report_service()
    try:
        if format == "json":
            report_service.export_json(student_id, output_path)
        elif format == "pdf":
            report_service.export_pdf(student_id, output_path)
        else:
            report_service.export_excel(student_id, output_path)
    except RuntimeError as e:
        # Raised by export_pdf()/export_excel() when the optional
        # dependency isn't installed - a configuration issue, not a
        # bug in the report itself.
        raise HTTPException(status_code=501, detail=str(e))

    return FileResponse(
        path=output_path,
        media_type=content_type,
        filename=output_path.name,
    )