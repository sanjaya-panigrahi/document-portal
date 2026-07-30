from datetime import datetime, timezone
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
    def __init__(self, data_dir: str = "data/single_document_chat", faiss_dir: str = "data/faiss_index"):
        try:
            self.logger = CustomLogger().get_logger(__name__)
            self.loader = ModelLoader()
            self.data_dir = Path(data_dir)
            self.faiss_dir = Path(faiss_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("SingleDocIngestion initialized successfully", data_dir=str(self.data_dir), faiss_dir=str(self.faiss_dir))
        except Exception as e:
            self.logger.error("Failed to initialize SingleDocIngestion", error=str(e))
            raise DocumentPortalException("Failed to initialize SingleDocIngestion", sys)
    
    def ingest_files(self, uploaded_files):
        try:
            documents = []
            for uploaded_file in uploaded_files:
                unique_filename = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
                temp_path = self.data_dir / unique_filename
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                self.logger.info("File saved in Ingestion successfully", filename=uploaded_file.name, temp_path=str(temp_path))
                loader = PyPDFLoader(str(temp_path))
                docs = loader.load()
                documents.extend(docs)
            self.logger.info("All files ingested successfully", total_documents=len(documents))
            return self._create_retrieval(documents)
        except Exception as e:
            self.logger.error("Failed to ingest files", error=str(e))
            raise DocumentPortalException("Failed to ingest files", sys)
    def _create_retrieval(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=300)
            chunks = splitter.split_documents(documents)
            self.logger.info("Documents split into chunks successfully", total_chunks=len(chunks))
            embeddings = self.loader.load_embeddings()
            
            vectorStore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            vectorStore.save_local(str(self.faiss_dir))
            self.logger.info("FAISS vector store created and saved successfully", faiss_dir=str(self.faiss_dir))
            
            retriever = vectorStore.as_retriever(search_kwargs={"k": 5})
            self.logger.info("Retriever created successfully")
            return retriever
        except Exception as e:
            self.logger.error("Failed to create retrieval", error=str(e))
            raise DocumentPortalException("Failed to create retrieval", sys)
        