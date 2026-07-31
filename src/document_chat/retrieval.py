import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentIntelligenceError
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType


class ConversationalRAG:
    """
    Conversational Retrieval-Augmented Generation (RAG) pipeline.

    What it does:
        1. Loads the selected LLM.
        2. Loads a FAISS vector index from disk.
        3. On each user query it:
           a. Rewrites the question to be standalone (removing dependency on chat history).
           b. Retrieves the top-k most relevant document chunks from FAISS.
           c. Passes those chunks + the question to the LLM to generate an answer.

    Two-step usage:
        rag = ConversationalRAG(session_id="abc", model_key="google")
        rag.load_retriever_from_faiss("faiss_index/abc", k=5)
        answer = rag.invoke("What is the main topic?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], model_key: Optional[str] = None, retriever=None):
        """
        Initialises the RAG pipeline.

        Args:
            session_id:  Unique ID for this chat session. Used to locate the correct FAISS index folder.
            model_key:   Which LLM to use (e.g. "google", "openai"). Defaults to LLM_PROVIDER env var.
            retriever:   Optionally pass a pre-built retriever to skip load_retriever_from_faiss().
        """
        try:
            self.session_id = session_id
            self.model_key = model_key

            # Load LLM and prompts once
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id, model_key=self.model_key)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentIntelligenceError("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Loads a FAISS vector store from disk and wires up the LCEL retrieval chain.
        Must be called before invoke() if no retriever was passed to __init__.

        Args:
            index_path:    Path to the FAISS index folder (e.g. "faiss_index/session123").
            k:             Number of top similar chunks to retrieve per query (default 5).
            index_name:    Name used when saving the index with FAISS.save_local() (default "index").
            search_type:   FAISS search strategy — "similarity" or "mmr" (default "similarity").
            search_kwargs: Extra kwargs passed to as_retriever(), e.g. {"fetch_k": 20} for MMR.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # ok if you trust the index
            )

            if search_kwargs is None:
                search_kwargs = {"k": k}

            self.retriever = vectorstore.as_retriever(
                search_type=search_type, search_kwargs=search_kwargs
            )
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentIntelligenceError("Loading error in ConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """
        Runs the full RAG pipeline for a single user question.

        Args:
            user_input:   The question the user typed.
            chat_history: Previous messages in this conversation (LangChain BaseMessage list).
                          Pass an empty list [] for the first turn.

        Returns:
            The LLM’s answer as a plain string.
        """
        try:
            if self.chain is None:
                raise DocumentIntelligenceError(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."
            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentIntelligenceError("Invocation error in ConversationalRAG", sys)

    # ---------- Internals ----------

    def _load_llm(self):
        """Internal: loads the LLM using ModelLoader with the stored model_key."""
        try:
            llm = ModelLoader().load_llm(model_key=self.model_key)
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id, model_key=self.model_key)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentIntelligenceError("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        """Internal: joins retrieved document chunks into a single context string for the prompt."""
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        """
        Internal: assembles the three-stage LCEL chain.
          Stage 1 — Rewrite the question to remove dependency on chat history.
          Stage 2 — Retrieve matching document chunks from FAISS.
          Stage 3 — Answer using the retrieved context + original question.
        """
        try:
            if self.retriever is None:
                raise DocumentIntelligenceError("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentIntelligenceError("Failed to build LCEL chain", sys)
