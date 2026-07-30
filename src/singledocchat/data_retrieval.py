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
            self._history_store = {}
            self.contextualize_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            self.history_aware_retriever= create_history_aware_retriever(
                self.llm, self.retriever, self.contextualize_prompt
            )
            self.logger.info("ConversationalRAG initialized successfully", session_id=self.session_id)
            self.qa_chain = create_stuff_documents_chain(
                self.llm,
                self.qa_prompt
            )
            self.rag_chain = create_retrieval_chain(
                self.history_aware_retriever,
                self.qa_chain   
            )
            
            self.logger.info("RAG chain created successfully", session_id=self.session_id)
            
            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            self.logger.info("RunnableWithMessageHistory created successfully", session_id=self.session_id)
        except Exception as e:
            self.logger.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Failed to initialize ConversationalRAG", sys)
    
    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            self.logger.info("LLM loaded successfully", class_name=llm.__class__.__name__)
            return llm
        except Exception as e:
            self.logger.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("Failed to load LLM", sys)
        
    def _get_session_history(
        self,
        session_id: str,
    ) -> BaseChatMessageHistory:
        if session_id not in self._history_store:
            self._history_store[session_id] = ChatMessageHistory()

        return self._history_store[session_id]
    
    def load_retriever_from_faiss(self, index_path:str):
        try:
            embeddings = self.loader.load_embeddings()
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index file not found at {index_path}")
            vector_store = FAISS.load_local(index_path, embeddings)
            self.logger.info("Retriever loaded from FAISS successfully", index_path=index_path)
            return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        except Exception as e:
            self.logger.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Failed to load retriever from FAISS", sys)
            
    def invoke(self, user_input:str)-> str:
        try:
            response = self.chain.invoke(
                {"input": user_input},
                config = {"configurable":{"session_id": self.session_id}}
            )
            
            answer = response.get("answer", "No answer.")
            if not answer:
                self.logger.warning("No answer returned from RAG chain", session_id=self.session_id)
            self.logger.info("Answer returned from RAG chain", answer=answer)
            
            return answer
            
        except Exception as e:
            self.logger.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Failed to invoke ConversationalRAG", sys)
