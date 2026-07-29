import sys
from pathlib import Path
import fitz
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class DocumentIngestion:
    
    def __init__(self, base_dir:str="data/document_compare"):
        """
        Initialize the DocumentIngestion class.
        """
        self.logger = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def delete_existing_files(self):
        """
        Delete existing files related to the document comparison session.
        """
        try:
            if self.base_dir.exists() and self.base_dir.is_dir():
                for file in self.base_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                        self.logger.info("Deleted file", path=str(file))
                self.logger.info("Directory Cleaned", directory = str(self.base_dir))
        except Exception as e:
            self.logger.error("Failed to delete existing files", error=str(e))
            raise DocumentPortalException("Failed to delete existing files", sys)
    
    def save_uploaded_files(self, reference_file, actual_file):
        """
        Save the uploaded files for comparison.
        """
        try:
            self.delete_existing_files()
            self.logger.info("Existing file deleted successfully", session_id=self.base_dir.name)
            
            if not reference_file.name.endswith('.pdf') or not actual_file.name.endswith('.pdf'):
                raise ValueError("Uploaded files must be PDFs", sys)

            ref_path = self.base_dir / reference_file.name
            act_path = self.base_dir / actual_file.name

            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())
            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())
            self.logger.info("Uploaded files saved successfully", reference_file=reference_file.name, 
                             actual_file=actual_file.name, session_id=self.base_dir.name)
            return ref_path, act_path
        except Exception as e:
            self.logger.error("Failed to save uploaded files", error=str(e))
            raise DocumentPortalException("Failed to save uploaded files", sys)
    
    def read_pdf(self, pdf_path:Path) -> str:
        """
        Read the content of the uploaded PDF files.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted and cannot be read: {pdf_path.name}")
                
                all_text = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        all_text.append(f"\n--- page {page_num} ---\n{text}")
                self.logger.info("PDF read successfully", file=pdf_path.name, session_id=self.base_dir.name)
                return "\n".join(all_text)
        except Exception as e:
            self.logger.error("Failed to read PDF", error=str(e))
            raise DocumentPortalException("Failed to read PDF", sys)
    
    def combine_documents(self, reference_file:Path, actual_file:Path) -> str:
        """
        Combine the content of the reference and actual PDF files.
        """
        try:
            content_dict = {}
            doc_parts = []

            for filename in (reference_file, actual_file):
                if filename.is_file() and filename.suffix.lower() == ".pdf":
                    content_dict[filename.name] = self.read_pdf(filename)
                else:
                    raise ValueError(f"Invalid PDF path: {filename}")
            
            for filename, content in content_dict.items():
                doc_parts.append(f"Documents :  {filename}\n{content}")
            
            combined_text = "\n\n".join(doc_parts)
            self.logger.info("Documents Combine", count = len(doc_parts))
            return combined_text
        except Exception as e:
            self.logger.error("Failed to combine documents", error=str(e))
            raise DocumentPortalException("Failed to combine documents", sys)
    
    
