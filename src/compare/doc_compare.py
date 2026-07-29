import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser

class DocumentComparatorLLM:
    def __init__(self):
        """
        Initialize the DocumentComparatorLLM class.
        """
        load_dotenv()
        self.logger = CustomLogger().get_logger(__name__)
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(llm=self.llm, parser=self.parser)
        self.prompt = PROMPT_REGISTRY["document_comparison"]
        # OutputFixingParser already wraps self.parser; chaining both causes type mismatches.
        self.chain = self.prompt | self.llm | self.parser
        
        self.logger.info("DocumentComparatorLLM initialized successfully")
    
    def compare_documents(self, combined_docs:str) -> pd.DataFrame:
        """
        Compare the content of the uploaded PDF files.
        """
        try:
            input = {
                    "combined_docs": combined_docs,
                    "format_instructions": self.parser.get_format_instructions()
                    }
            self.logger.info("Invoking document comparison chain", input_keys=list(input.keys()))
            response = self.chain.invoke(input)
            self.logger.info("Document comparison completed successfully", response=response)
            return self._format_response(response)
        except Exception as e:
            self.logger.error("Failed to compare documents", error=str(e))
            raise DocumentPortalException("Failed to compare documents", sys)
    
    def _format_response(self, response:list[dict]) -> pd.DataFrame:
        """
        Format the response from the document comparison.
        """
        try:
            df = pd.DataFrame(response)
            self.logger.info("Response formatted into DataFrame", shape=df.shape)
            return df
        except Exception as e:
            self.logger.error("Failed to format response", error=str(e))
            raise DocumentPortalException("Failed to format response", sys)