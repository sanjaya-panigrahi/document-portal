from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# Prompt Library
# All LangChain prompt templates used in this application are defined here
# and registered in PROMPT_REGISTRY so any module can look them up by name.
# ---------------------------------------------------------------------------

# Prompt for document analysis
# Instructs the LLM to extract structured metadata (Title, Author, Summary, etc.)
# and return it as valid JSON matching the Metadata schema.
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

# Prompt for document comparison
# Instructs the LLM to compare two PDF documents page-by-page and list changes.
# Input variables: {combined_docs}, {format_instruction}
document_comparison_prompt = ChatPromptTemplate.from_template("""
You will be provided with content from two PDFs. Your tasks are as follows:

1. Compare the content in two PDFs
2. Identify the difference in PDF and note down the page number 
3. The output you provide must be page wise comparison content 
4. If any page do not have any change, mention as 'NO CHANGE' 

Input documents:

{combined_docs}

Your response should follow this format:

{format_instruction}
""")

# Prompt for contextual question rewriting
# Rewrites the latest user question to be self-contained (no reference to chat history).
# Required by the RAG pipeline so the retriever gets a clean, standalone query.
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt for answering based on context
# Answers the user question strictly from the retrieved document context.
# Responds with "I don't know" if the answer is not found in the context.
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Central registry — maps string keys to prompt templates.
# Used by DocumentAnalyzer, DocumentComparatorLLM, and ConversationalRAG
# via PROMPT_REGISTRY[PromptType.XXX.value].
PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}
