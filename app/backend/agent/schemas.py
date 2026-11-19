
from pydantic import BaseModel, ConfigDict


class RAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    source_ids: list[str]
