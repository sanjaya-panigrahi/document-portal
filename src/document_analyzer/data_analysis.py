import os
import sys
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentIntelligenceError
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY # type: ignore

class DocumentAnalyzer:
    """
    Extracts structured metadata from a PDF document using an LLM.

    What it does:
        - Loads the selected LLM via ModelLoader.
        - Uses a JSON output parser to enforce a strict response schema (Metadata model).
        - If the LLM returns malformed JSON, OutputFixingParser automatically retries and repairs it.
        - Runs a single LangChain chain: prompt → LLM → parser.

    Usage:
        analyzer = DocumentAnalyzer(model_key="google")
        result = analyzer.analyze_document(text)
    """
    def __init__(self, model_key: str | None = None):
        """
        Sets up the LLM, parsers, and prompt template.

        Args:
            model_key: Which LLM to use (e.g. "google", "openai", "groq").
                       If None, falls back to the LLM_PROVIDER env variable.
        """
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm(model_key=model_key)

            # JsonOutputParser enforces the Metadata pydantic schema on the LLM response.
            # OutputFixingParser wraps it to auto-repair bad JSON before raising an error.
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

            self.prompt = PROMPT_REGISTRY["document_analysis"]

            log.info("DocumentAnalyzer initialized successfully", model_key=model_key)
            
            
        except Exception as e:
            log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentIntelligenceError("Error in DocumentAnalyzer initialization", sys)
        
        
    
    def analyze_document(self, document_text: str) -> dict:
        """
        Runs the LLM chain on the given document text and returns structured metadata.

        Args:
            document_text: Raw text extracted from the PDF (page-by-page).

        Returns:
            A dict matching the Metadata schema — Title, Author, Summary,
            PageCount, SentimentTone, DateCreated, Language, etc.
        """
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            
            log.info("Meta-data analysis chain initialized")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            log.info("Metadata extraction successful", keys=list(response.keys()))
            
            return response

        except Exception as e:
            log.error("Metadata analysis failed", error=str(e))
            raise DocumentIntelligenceError("Metadata extraction failed",sys)
        
    
