from typing import Optional
from pydantic import BaseModel


class PreprocessData(BaseModel):
    images_processed: int
    successful: int
    failed: int
    csv_file: str
    normalized_csv_file: Optional[str] = None


class PreprocessResponse(BaseModel):
    success: bool
    message: str
    data: PreprocessData