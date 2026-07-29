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
        pass
    
    def compare_documents(self):
        """
        Compare the content of the uploaded PDF files.
        """
        pass
    
    def _fromat_response(self):
        """
        Format the response from the document comparison.
        """
        pass