import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

log = CustomLogger().get_logger(__name__)
class ModelLoader:
    """A utility class to load and manage models, embeddings, 
    and configurations for the document portal application.
    Attributes:
        config (dict): Configuration settings loaded from a YAML file.
    """
    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        log.info("Configuration loaded successfully", config_keys=list(self.config.keys()))
    
    def _validate_env(self):
        """
        Validate the required environment variables.
        """
        required_vars = ["GOOGLE_API_KEY", "GROQ_API_KEY"]
        self.api_keys = {key: os.getenv(key) for key in required_vars}
        
        missing = [k for k, v in self.api_keys.items() if not v]
        if missing:
            log.error("Missing required environment variables", missing_vars=missing)
            raise DocumentPortalException("Missing required environment variables", sys)
        
    def load_embeddings(self):
        """
        Load the embeddings model.
        Returns:
            An embeddings model instance.
        """
        try:
            # Placeholder for actual embeddings loading logic
            log.info("Loading embeddings model")
            model_name = self.config["embeddings_model"]["model_name"]
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("Failed to load embeddings model", error=str(e))
            raise DocumentPortalException("Failed to load embeddings model", sys)
    
    def load_llm(self):
        """Load and return LLM model"""
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "groq")
        
        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider_key=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 2048)
        
        log.info("Loading LLM model", provider=provider, model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        
        if provider == "groq":
            llm = ChatGroq(
                model=model_name, 
                temperature=temperature, 
                max_tokens=max_tokens
                )
            return llm
        elif provider == "google":
            llm = ChatGoogleGenerativeAI(
                model=model_name, 
                temperature=temperature, 
                max_output_tokens=max_tokens
                )
            return llm
        else:   
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
    
if __name__ == "__main__":
    try:
        model_loader = ModelLoader()
        log.info("ModelLoader initialized successfully")
        
        embeddings = model_loader.load_embeddings()
        log.info("Embeddings model loaded successfully", embeddings_model=embeddings)
        embeddings_result = embeddings.embed_query("Hello, how are you?")
        log.info("Embeddings invocation result", result=embeddings_result)
        
        llm = model_loader.load_llm()
        log.info("LLM model loaded successfully", llm_model=llm)
        
        result = llm.invoke("Hello, how are you?")
        log.info("LLM invocation result", result=result)
    except DocumentPortalException as e:
        log.error("Failed to initialize ModelLoader", error=str(e))
    
    
    