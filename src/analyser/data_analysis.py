import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
from model.models import Metadata
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from prompt.prompt_library import *


class DocumentAnalyser:
    """Class to handle document analysis and processing."""

    def __init__(self):
        self.logger = CustomLogger().get_logger(__name__)
        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()
            
            # Initialize output parsers
            self.parser = JsonOutputParser(pydantic_object= Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(llm=self.llm, parser=self.parser)
            
            self.prompt = prompt
            self.logger.info("DocumentAnalyser initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize DocumentAnalyser", error=str(e))
            raise DocumentPortalException("Failed to initialize DocumentAnalyser", sys)
    
    def analyse_document(self, document_text) -> dict:
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            self.logger.info("Document analysis chain initialized")
            
            response = chain.invoke({
                "document_text": document_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            self.logger.info("Document analysis completed successfully", keys = list(response.keys()))
            return response
        except Exception as e:
            self.logger.error("Failed to analyse document", error=str(e))
            raise DocumentPortalException("Failed to analyse document", sys)


if __name__ == "__main__":
    analyser = DocumentAnalyser()
    analyser.analyse_document("Sample document text for analysis.")