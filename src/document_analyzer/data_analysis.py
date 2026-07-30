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
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization.
    """
    def __init__(self, model_key: str | None = None):
        try:
            self.loader=ModelLoader()
            self.llm=self.loader.load_llm(model_key=model_key)
            
            # Prepare parsers
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            
            self.prompt = PROMPT_REGISTRY["document_analysis"]
            
            log.info("DocumentAnalyzer initialized successfully", model_key=model_key)
            
            
        except Exception as e:
            log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentIntelligenceError("Error in DocumentAnalyzer initialization", sys)
        
        
    
    def analyze_document(self, document_text:str)-> dict:
        """
        Analyze a document's text and extract structured metadata & summary.
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
        
    
