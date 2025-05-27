from pydantic import BaseModel, HttpUrl
from typing import List, Optional

# Request model for ingesting documents
class IngestRequest(BaseModel):
    """
    Represents a request to ingest documents from a list of URLs.
    """
    urls: List[HttpUrl]  # List of URLs to fetch documents from
    chunk_documents: bool = True  # Whether to chunk documents before embedding

# Response model for document ingestion
class IngestResponse(BaseModel):
    """
    Represents the response after attempting to ingest documents.
    """
    message: str  # A message indicating the status of the ingestion
    ingested_urls: List[str]  # List of URLs that were successfully ingested
    failed_urls: List[str] = [] # List of URLs that failed to ingest
    errors: List[str] = [] # List of error messages for failed URLs

# Request model for a chat message
class ChatRequest(BaseModel):
    """
    Represents a user's chat message.
    """
    message: str  # The user's query/message

# Response model for a chat message
class ChatResponse(BaseModel):
    """
    Represents the chatbot's response to a message.
    """
    response: str  # The generated response from the chatbot
    source_documents: List[str]  # A list of source document snippets used for the response
