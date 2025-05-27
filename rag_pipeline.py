import os
import asyncio
import httpx # Still needed for fallback web scraping
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec, Index
 # Import ServerlessSpec
from litellm import aembedding, acompletion
import litellm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from time import sleep
litellm.drop_params = True # This line is the fix!


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Configuration from Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini/text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Pinecone Serverless Specifics (Optional, ensure these are set in .env if used)
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws") # e.g., 'aws', 'gcp', 'azure'
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1") # e.g., 'us-east-1', 'eu-west-1'


# --- Pinecone Initialization ---
pc = None
pinecone_index: Index = None
EMBEDDING_DIMENSION = 1536 # Default for 'gemini/text-embedding-004'. Adjust if using a different model.

# Define the mapping for local content to bypass web scraping for specific URLs
LOCAL_DOCUMENT_MAP = {
    "https://help.typeform.com/hc/en-us/articles/23541138531732-Create-multi-language-forms": "data/multi_language_forms.txt",
    "https://help.typeform.com/hc/en-us/articles/27703634781076-Add-a-Multi-Question-Page-to-your-form": "data/multi_question_page.txt"
}

async def initialize_pinecone():
    """
    Initializes the Pinecone client and ensures the index exists.
    It supports creating either PodSpec or ServerlessSpec indexes based on environment variables.
    """
    global pc, pinecone_index
    if not PINECONE_API_KEY or not PINECONE_CLOUD or not PINECONE_REGION or not PINECONE_INDEX_NAME:
        logging.error("Pinecone environment variables (API_KEY, ENVIRONMENT, INDEX_NAME) not fully set. Cannot initialize Pinecone.")
        return False
    
    # Determine which Pinecone spec to use for index creation
    # We use ServerlessSpec if PINECONE_CLOUD and PINECONE_REGION are both provided.
    use_serverless_spec = PINECONE_CLOUD and PINECONE_REGION
    if use_serverless_spec:
        logging.info("Attempting to use ServerlessSpec for Pinecone index creation.")
        # Ensure serverless specific variables are actually set if we intend to use them
        if not PINECONE_CLOUD or not PINECONE_REGION:
            logging.error("PINECONE_CLOUD or PINECONE_REGION missing for ServerlessSpec.")

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logging.info(f"Checking for Pinecone index: {PINECONE_INDEX_NAME}")
        
        existing_indexes = pc.list_indexes().names()
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            logging.info(f"Index '{PINECONE_INDEX_NAME}' not found. Creating...")
            
            # Define the spec based on the determination above
            if use_serverless_spec:
                spec_to_use = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
                logging.info(f"Creating Pinecone serverless index: {PINECONE_INDEX_NAME} (Cloud: {PINECONE_CLOUD}, Region: {PINECONE_REGION})")
            
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric='cosine',
                spec=spec_to_use
            )
            logging.info(f"Pinecone index '{PINECONE_INDEX_NAME}' created successfully.")
        else:
            logging.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        return True
    except Exception as e:
        logging.error(f"Error initializing Pinecone: {e}")
        return False

# --- Document Processing Functions ---

async def _fetch_document(url: str) -> str:
    """
    Fetches the content of a given URL.
    Prioritizes loading from a local file if the URL is mapped in LOCAL_DOCUMENT_MAP.
    Applies BeautifulSoup parsing to both local and web-fetched HTML content.
    """
    raw_content = ""
    # Check if the URL is mapped to a local file
    local_file_path = LOCAL_DOCUMENT_MAP.get(url)
    if local_file_path:
        try:
            # Construct absolute path for the local file
            # Assumes 'data/' is relative to the directory where rag_pipeline.py resides
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(base_dir, local_file_path)
            logging.info(f"Loading content for {url} from local file: {full_path}")
            with open(full_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except FileNotFoundError:
            logging.error(f"Local document file not found: {full_path}. Please ensure 'data/' directory and files exist.")
            raise # Re-raise if the local file isn't there
        except Exception as e:
            logging.error(f"Error reading local document {full_path}: {e}")
            raise
    else:
        # Fallback to web scraping for other URLs
        try:
            async with httpx.AsyncClient() as client:
                # Add a User-Agent header to mimic a browser, which can help bypass 403 errors
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36"
                }
                logging.info(f"Fetching content for {url} from web.")
                response = await client.get(url, timeout=30.0, headers=headers)
                response.raise_for_status()  # Raise an exception for bad status codes (like 4xx, 5xx)
                raw_content = response.text
        except httpx.RequestError as e:
            logging.error(f"Network error fetching URL {url}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error fetching document from {url}: {e}")
            raise

    # Apply BeautifulSoup parsing to the raw_content (whether from local file or web)
    try:
        soup = BeautifulSoup(raw_content, 'html.parser')

        # Remove script, style, and other non-content tags for cleaner text extraction
        for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]):
            script_or_style.extract()

        # Get text and clean it up by stripping extra whitespace
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception as e:
        logging.error(f"Error parsing HTML content from {url}: {e}")
        raise

def _chunk_document(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Chunks a given text into smaller, overlapping segments.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Chunked document into {len(chunks)} segments.")
    return chunks

async def _embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of text strings using LiteLLM.
    """
    try:
        response = await aembedding(
            model=EMBEDDING_MODEL,
            input=texts,
            api_key=OPENAI_API_KEY,
        )
        print(response.data[0].keys())
        embeddings = [item['embedding'] for item in response.data]
        logging.info(f"Generated {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        logging.error(f"Error generating embeddings with LiteLLM: {e}")
        raise

async def ingest_documents(urls: list[str], chunk_documents: bool) -> tuple[list[str], list[str], list[str]]:
    """
    Fetches, processes (optionally chunks), embeds, and upserts documents to Pinecone.
    """
    if not pinecone_index:
        success = await initialize_pinecone()
        if not success:
            # Ensure failed_urls are strings if Pinecone initialization fails
            stringified_urls = [str(url) for url in urls]
            return [], stringified_urls, ["Pinecone not initialized. Check API keys and environment."]

    ingested_urls = []
    failed_urls = []
    errors = []

    for url in urls:
        source_url_str = str(url) # Convert HttpUrl to string for consistent logging and metadata
        try:
            logging.info(f"Processing URL: {source_url_str}")
            document_text = await _fetch_document(source_url_str)
            
            chunks_to_embed = []
            if chunk_documents:
                chunks_to_embed = _chunk_document(document_text, CHUNK_SIZE, CHUNK_OVERLAP)
            else:
                chunks_to_embed = [document_text] # Treat entire document as one chunk

            if not chunks_to_embed:
                logging.warning(f"No content or chunks generated for {source_url_str}. Skipping.")
                failed_urls.append(source_url_str)
                errors.append(f"No content found or generated for {source_url_str}")
                continue
            sleep(5.0)
            embeddings = await _embed_texts(chunks_to_embed)

            vectors_to_upsert = []
            for i, (chunk, embed) in enumerate(zip(chunks_to_embed, embeddings)):
                # Create a unique ID for each vector based on URL and chunk index
                # Replace special characters from URL for a valid Pinecone ID
                vector_id = f"{source_url_str.replace('.', '_').replace('/', '_').replace(':', '_').replace('?', '_').replace('=', '_').replace('&', '_').replace('-', '_')}_chunk_{i}"
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embed,
                    "metadata": {
                        "url": source_url_str,
                        "chunk_index": i,
                        "text": chunk
                    }
                })
            
            # Upsert in batches (Pinecone recommends batches of ~100 vectors for performance)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                pinecone_index.upsert(vectors=batch)
                logging.info(f"Upserted batch of {len(batch)} vectors for {source_url_str}.")

            ingested_urls.append(source_url_str)
            logging.info(f"Successfully ingested {source_url_str}")

        except httpx.RequestError as e:
            logging.error(f"Network error ingesting {source_url_str}: {e}")
            failed_urls.append(source_url_str)
            errors.append(f"Network error for {source_url_str}: {e}")
        except FileNotFoundError: # Catch if local file mapping exists but file is missing
            logging.error(f"Local file not found for {source_url_str}. Skipping ingestion.")
            failed_urls.append(source_url_str)
            errors.append(f"Local file not found for {source_url_str}. Please check data directory.")
        except Exception as e:
            logging.error(f"Failed to ingest {source_url_str}: {e}")
            failed_urls.append(source_url_str)
            errors.append(f"Failed to ingest {source_url_str}: {e}")
    
    return ingested_urls, failed_urls, errors

async def get_rag_response(query: str) -> tuple[str, list[str]]:
    """
    Retrieves relevant context from Pinecone based on a query and generates a response using an LLM.
    """
    if not pinecone_index:
        logging.error("Pinecone not initialized. Cannot retrieve RAG response.")
        return "Error: Pinecone not initialized. Please ensure the service is running and configured correctly.", []

    try:
        # 1. Embed the query
        query_embedding_response = await _embed_texts([query])
        query_embedding = query_embedding_response[0]

        # 2. Retrieve relevant documents from Pinecone
        # Fetch top k results
        top_k_results = pinecone_index.query(
            vector=query_embedding,
            top_k=3, # Retrieve top 3 relevant documents
            include_metadata=True
        )

        context_texts = [match.metadata['text'] for match in top_k_results.matches if match.score > 0.7] # Filter by score
        source_documents_urls = list(set([match.metadata['url'] for match in top_k_results.matches if match.score > 0.7]))
        
        if not context_texts:
            logging.info("No relevant context found in Pinecone. Responding without RAG.")
            # If no context, provide a direct LLM response or a default message
            llm_response = await acompletion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": query}],
                api_key=OPENAI_API_KEY
            )
            return llm_response.choices[0].message.content, []
        
        context_str = "\n".join(context_texts)
        logging.info(f"Retrieved context:\n{context_str[:200]}...") # Log first 200 chars

        # 3. Formulate prompt for the LLM
        prompt = f"""
        You are a helpful and friendly assistant that answers questions based on the provided context in a conversational and succinct way.
        If the answer is not in the context, politely state that you don't have enough information.

        Context:
        {context_str}

        Question: {query}
        """

        # 4. Get response from LLM using LiteLLM
        llm_response = await acompletion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            api_key=OPENAI_API_KEY
        )
        response_content = llm_response.choices[0].message.content
        logging.info(f"Generated LLM response.")

        return response_content, source_documents_urls

    except Exception as e:
        logging.error(f"Error getting RAG response: {e}")
        return f"An error occurred while generating the response: {e}", []

# This function is used by FastAPI's lifespan context manager
async def startup_event():
    """
    Function to be run on application startup to initialize Pinecone.
    """
    await initialize_pinecone()