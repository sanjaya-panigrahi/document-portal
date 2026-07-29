import sys
import os
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import Metadata, SummaryResponse, PromptType

class ConversationalRAG:
    def __init__(self, session_id:str, retriever) -> None:
        try:
            self.logger = CustomLogger().get_logger(__name__)
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()
            self.session_id = session_id
            self.retriever = retriever
        except Exception as e:
            self.logger.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Failed to initialize ConversationalRAG", sys)
    
    def _load_llm(slef):
        try:
            pass
        except Exception as e:
            self.logger.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("Failed to load LLM", sys)
        
    def _get_session_history(self, session_id:str):
        try:
            pass
        except Exception as e:
            self.logger.error("Failed to get session history", error=str(e))
            raise DocumentPortalException("Failed to get session history", sys)
    
    def load_retriever_from_faiss(self):
        try:
            pass
        except Exception as e:
            self.logger.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Failed to load retriever from FAISS", sys)
    
    def invoke(self):
        try:
            pass
        except Exception as e:
            self.logger.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Failed to invoke ConversationalRAG", sys)