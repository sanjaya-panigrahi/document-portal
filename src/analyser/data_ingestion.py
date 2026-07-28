import os
import sys
import fitz
import uuid
from datetime import datetime
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class DocumentHandler:
    """Class to handle document ingestion and processing."""

    def __init__(self, data_dir = None, session_id = None):
        try:
            self.logger = CustomLogger().get_logger(__name__)
            self.data_dir = data_dir or os.getenv(
                "DATA_STORAGE_PATH",
                os.path.join(os.getcwd(), "data", "document_analysis")
            )
            self.session_id = session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Create base session directory
            self.session_path = os.path.join(self.data_dir, self.session_id)
            os.makedirs(self.session_path, exist_ok=True)
            
            self.logger.info("DocumentHandler initialized", session_id=self.session_id, session_path=self.session_path)
        except Exception as e:
            self.logger.error("Failed to initialize DocumentHandler", error=str(e))
            raise DocumentPortalException("Failed to initialize DocumentHandler", sys)

    def save_pdf(self, uploaded_file):
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith('.pdf'):
                raise DocumentPortalException("Uploaded file is not a PDF", sys)
            
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            self.logger.info("PDF saved successfully", file = filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            self.logger.error("Failed to save PDF", error=str(e))
            raise DocumentPortalException("Failed to save PDF", sys)
    
    def read_pdf(self, pdf_path):
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text_chunks.append(f"\n--- page {page_num} ---\n{page.get_text()}")
            text = "\n".join(text_chunks)
            self.logger.info("PDF read successfully", file = os.path.basename(pdf_path), session_id=self.session_id)
            return text
        except Exception as e:
            self.logger.error("Failed to read PDF", error=str(e))
            raise DocumentPortalException("Failed to read PDF", sys)
        
if __name__ == "__main__":
    handler = DocumentHandler()
    from io import BytesIO
    from pathlib import Path
    pdf_path = r"/Users/sanjaya-panigrahi/Projects/JOB PREP/Project_Repo/document-portal/data/document_analysis/NIPS-2017-attention-is-all-you-need-Paper.pdf"
    
    class DummyFile:
        def __init__(self, file_path):
            self._file_path = file_path
            self.name = Path(file_path).name
        def getbuffer(self):
            return open(self._file_path, "rb").read()
    
    dummy_pdf = DummyFile(pdf_path)
    handler = DocumentHandler(session_id="test_session")
    try:
        save_path = handler.save_pdf(dummy_pdf)
        print(f"PDF saved at: {save_path}")
        
        content = handler.read_pdf(save_path)
        print(f"PDF content length: {len(content)} characters")
    except Exception as e:
        handler.logger.error("Error during document handling", error=str(e))