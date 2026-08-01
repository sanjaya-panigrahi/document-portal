import sys
from dotenv import load_dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentIntelligenceError
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import SummaryResponse,PromptType

class DocumentComparatorLLM:
    """
    Compares two PDF documents using an LLM and returns page-wise differences.

    What it does:
        - Takes combined text from two documents (reference vs actual).
        - Sends it through a comparison prompt to the LLM.
        - Parses the response into a list of { Page, Changes } rows.
        - Returns a pandas DataFrame so results can be displayed as a table.

    Usage:
        comp = DocumentComparatorLLM(model_key="openai")
        df = comp.compare_documents(combined_text)
    """
    def __init__(self, model_key: str | None = None):
        """
        Sets up the LLM, parsers, and comparison prompt.

        Args:
            model_key: Which LLM to use. Falls back to LLM_PROVIDER env var if None.
        """
        load_dotenv()
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm(model_key=model_key)
        # SummaryResponse is a list of ChangeFormat rows — each has a Page and Changes field.
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        # Full chain: comparison prompt → LLM → JSON parser
        self.chain = self.prompt | self.llm | self.parser
        log.info("DocumentComparatorLLM initialized", model=self.llm, model_key=model_key)

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        """
        Sends the combined document text to the LLM and returns differences as a DataFrame.

        Args:
            combined_docs: Output from DocumentComparator.combine_documents() —
                           reference and actual PDFs concatenated with section headers.

        Returns:
            A pandas DataFrame with columns: Page, Changes.
        """
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions()
            }

            log.info("Invoking document comparison LLM chain")
            response = self.chain.invoke(inputs)
            log.info("Chain invoked successfully", response_preview=str(response)[:200])
            return self._format_response(response)
        except Exception as e:
            log.error("Error in compare_documents", error=str(e))
            raise DocumentIntelligenceError("Error comparing documents", sys)

    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame:  # type: ignore
        """
        Converts the parsed LLM response (list of dicts) into a pandas DataFrame.

        Args:
            response_parsed: List of { "Page": "...", "Changes": "..." } dicts.
        """
        try:
            df = pd.DataFrame(response_parsed)
            return df
        except Exception as e:
            log.error("Error formatting response into DataFrame", error=str(e))
            DocumentIntelligenceError("Error formatting response", sys)
