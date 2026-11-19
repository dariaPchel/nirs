from pydantic import BaseModel


class AnalysisResult(BaseModel):
    document_id: str | None
    description: str
