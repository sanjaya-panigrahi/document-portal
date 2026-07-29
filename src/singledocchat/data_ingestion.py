import uuid
from pathlib import Path
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader

class SingleDocIngestion:
    def __init__(self):
        try:
            self.logger = CustomLogger().get_logger(__name__)
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()
        except Exception as e:
            self.logger.error("Failed to initialize SingleDocIngestion", error=str(e))
            raise DocumentPortalException("Failed to initialize SingleDocIngestion", sys)
    
    def ingest_files(self):
        try:
            pass
        except Exception as e:
            self.logger.error("Failed to ingest files", error=str(e))
            raise DocumentPortalException("Failed to ingest files", sys)
    
    def _create_retrieval(self):
        try:
            pass
        except Exception as e:
            self.logger.error("Failed to create retrieval", error=str(e))
            raise DocumentPortalException("Failed to create retrieval", sys)
        