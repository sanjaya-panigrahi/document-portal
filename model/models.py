from pydantic import BaseModel, RootModel
from typing import List, Union
from enum import Enum


class Metadata(BaseModel):
    """
    Schema for the structured output returned by DocumentAnalyzer.
    The LLM is instructed to fill every field from the document content.
    PageCount can be an int or the string "Not Available" when not found.
    """
    Summary: List[str]              # Key points / bullet summary of the document
    Title: str                      # Document title
    Author: List[str]               # List of authors
    DateCreated: str                # Original creation date
    LastModifiedDate: str           # Date of last modification
    Publisher: str                  # Publisher name
    Language: str                   # Language the document is written in
    PageCount: Union[int, str]      # Total pages; "Not Available" if unknown
    SentimentTone: str              # Overall tone: positive / neutral / negative


class ChangeFormat(BaseModel):
    """
    A single page-level difference found during document comparison.
    Used as elements inside SummaryResponse.
    """
    Page: str       # Page number where the change was detected
    Changes: str    # Human-readable description of what changed


class SummaryResponse(RootModel[list[ChangeFormat]]):
    """
    Full comparison result — a list of per-page ChangeFormat entries.
    This is the schema the LLM must return from the document comparison prompt.
    """
    pass


class PromptType(str, Enum):
    """
    Enum keys used to look up prompts from PROMPT_REGISTRY in prompt_library.py.
    Using an enum prevents typos when referencing prompt names across the codebase.
    """
    DOCUMENT_ANALYSIS = "document_analysis"          # Prompt for metadata extraction
    DOCUMENT_COMPARISON = "document_comparison"      # Prompt for page-wise diff
    CONTEXTUALIZE_QUESTION = "contextualize_question" # Prompt to rewrite chat questions
    CONTEXT_QA = "context_qa"                        # Prompt to answer using retrieved context