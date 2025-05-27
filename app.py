from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from models import IngestRequest, IngestResponse, ChatRequest, ChatResponse
from rag_pipeline import ingest_documents, get_rag_response, startup_event
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define an async context manager for application lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes Pinecone on startup.
    """
    logging.info("Application startup event: Initializing Pinecone...")
    await startup_event()
    logging.info("Pinecone initialization complete (or skipped if already initialized).")
    yield # Application runs
    logging.info("Application shutdown event.")

# Create the FastAPI app instance with the defined lifespan
app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval Augmented Generation (RAG) chatbot using FastAPI, Pinecone, and LiteLLM.",
    version="1.0.0",
    lifespan=lifespan # Attach the lifespan context manager
)

@app.get("/", include_in_schema=False)
async def root():
    """
    Redirects to the OpenAPI documentation (Swagger UI).
    """
    return RedirectResponse(url="/docs")

@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_docs(request: IngestRequest):
    """
    Endpoint to ingest documents from provided URLs into the vector database.
    Documents will be fetched, optionally chunked, embedded, and then upserted to Pinecone.
    """
    logging.info(f"Received ingestion request for {len(request.urls)} URLs. Chunking: {request.chunk_documents}")
    ingested_urls, failed_urls, errors = await ingest_documents(request.urls, request.chunk_documents)
    
    if failed_urls:
        message = f"Ingestion completed with some failures. Successfully ingested {len(ingested_urls)} URLs, failed {len(failed_urls)}."
        logging.warning(message)
        return IngestResponse(
            message=message,
            ingested_urls=ingested_urls,
            failed_urls=failed_urls,
            errors=errors
        )
    else:
        message = f"Successfully ingested {len(ingested_urls)} URLs."
        logging.info(message)
        return IngestResponse(
            message=message,
            ingested_urls=ingested_urls
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Endpoint to interact with the RAG chatbot.
    The user's message will be used to retrieve relevant context from Pinecone,
    and then an LLM (via LiteLLM) will generate a response based on the context.
    """
    logging.info(f"Received chat message: '{request.message}'")
    response, source_documents = await get_rag_response(request.message)
    
    if "Error: Pinecone not initialized" in response:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response
        )
    
    return ChatResponse(response=response, source_documents=source_documents)

