from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union

class Metadata(BaseModel):
    """Pydantic model for metadata extracted from documents."""
    Summary: List[str] = Field(default_factory=list, description="A list of summary points extracted from the document.")
    Title: str
    Author: str
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    SentimentTone: str
    PageCount: Union[int,str]
    