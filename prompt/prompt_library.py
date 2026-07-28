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

prompt= ChatPromptTemplate.from_template(
    """
    You are a highly capable assistance trained to analyse and summarize documents. 
    Return Only a valid JSON matching the exact schemda below.
    
    {format_instructions}
    
    Analyze this document:
    {document_text}
    """
    )