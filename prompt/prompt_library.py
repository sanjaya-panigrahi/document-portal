from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template(
    """
    You are a highly capable assistance trained to analyse and summarize documents.
    Return Only a valid JSON matching the exact schemda below.

    {format_instructions}

    Analyze this document:
    {document_text}
    """
)
from langchain_core.prompts import ChatPromptTemplate

document_analysis_prompt= ChatPromptTemplate.from_template(
    """
    You are a highly capable assistance trained to analyse and summarize documents. 
    Return Only a valid JSON matching the exact schemda below.
    
    {format_instructions}
    
    Analyze this document:
    {document_text}
    """
    )

document_comparison_prompt = ChatPromptTemplate.from_template(
    """
    You will be provided with content from two PDFs. Your tasks are as follows:
    
    1. Compare the content in two PDFs
    2. Identify the difference in PDFs and note down the page number
    3. The output you provide must be page wise comparision content
    4. If any page do not have any changes, mentioned as  "NO CHANGE"
    
    Input documents:
    
    {combined_docs}
    
    Your response should follow this format:
    {format_instructions}
    
    """
    )

PROMPT_REGISTRY = {"document_analysis": document_analysis_prompt,
                   "document_comparison": document_comparison_prompt}