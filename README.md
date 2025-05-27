# RAG Chatbot with FastAPI, Pinecone, LiteLLM, and Docker/Kubernetes

This project implements a Retrieval Augmented Generation (RAG) chatbot API using FastAPI, Pinecone as a vector database, and LiteLLM for interacting with various Large Language Models (LLMs) and embedding models. The application is containerized with Docker and includes Helm chart configurations for easy deployment on Kubernetes.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Features](#features)
3.  [Prerequisites](#prerequisites)
4.  [Local Setup](#local-setup)
    * [Environment Variables](#environment-variables)
    * [Running Locally](#running-locally)
5.  [Docker Setup](#docker-setup)
    * [Building the Docker Image](#building-the-docker-image)
    * [Running the Docker Container](#running-the-docker-container)
6.  [Kubernetes Deployment with Helm](#kubernetes-deployment-with-helm)
    * [Helm Chart Structure](#helm-chart-structure)
    * [Deployment Steps](#deployment-steps)
7.  [API Endpoints](#api-endpoints)
8.  [Error Handling & Best Practices](#error-handling--best-practices)
9.  [Architectural Decisions & Trade-offs](#architectural-decisions--trade-offs)

---

## Project Overview

This solution provides an API-driven RAG chatbot. It allows users to ingest documents from URLs, which are then processed (optionally chunked), embedded, and stored in a Pinecone vector database. Subsequently, users can query the chatbot, and the system will retrieve relevant document snippets from Pinecone to augment the LLM's response, providing grounded and context-aware answers.

---

## Features

* **API Development**: Exposes endpoints for document ingestion and conversational chat.
* **RAG Pipeline**: Fetches, chunks, embeds documents, and retrieves relevant context for LLM responses.
* **Vector Database Integration**: Uses **Pinecone** for efficient similarity search of document embeddings.
* **LLM Agnostic (via LiteLLM)**: Leverages **LiteLLM** to support various LLM providers (e.g., OpenAI, Azure OpenAI, Cohere) with a unified interface.
* **Containerization**: Provides a **Dockerfile** for packaging the application into a portable Docker image.
* **Kubernetes Deployment**: Includes **Helm chart** configuration files for streamlined deployment on Kubernetes, ensuring local runnability is maintained.
* **Error Handling & Logging**: Implements basic error handling and logging for better observability.
* **Documentation**: Comprehensive `README.md` for setup and usage.

---

## Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.9+**
* **pip** (Python package installer)
* **Docker Desktop** (or Docker Engine)
* **kubectl** (Kubernetes command-line tool, if deploying to Kubernetes)
* **Helm 3+** (if deploying to Kubernetes)
* **Accounts/API Keys**:
    * An **OpenAI API Key** (or API key for your chosen LLM provider compatible with LiteLLM).
    * A **Pinecone API Key** and **Environment** (e.g., `us-east-1`).

---

## Local Setup

### Environment Variables

Create a file named `.env` in the root directory of the project. Fill in your actual API keys and desired configurations:

```dotenv
# .env
OPENAI_API_KEY="your_openai_api_key_here"

PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_CLOUD="your_pinecone_cloud_here" # e.g. aws, gcp, azure 
PINECONE_REGION="your_pinecone_region_here" # e.g., us-east-1 etc.
PINECONE_INDEX_NAME="typeform-rag-index"
EMBEDDING_MODEL="openai/text-embedding-ada-002" # Or another model supported by LiteLLM
LLM_MODEL="openai/gpt-3.5-turbo" # Or another model supported by LiteLLM

CHUNK_SIZE=1000 # Or whatever you prefer
CHUNK_OVERLAP=200 # Or whatever you prefer
```

## Running Locally
Clone the repository (if applicable) and navigate to the project root.
Install dependencies:
```Bash

pip install -r requirements.txt
```
Run the FastAPI application:
```Bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The `--reload` flag is useful for development as it automatically restarts the server on code changes.

### Access the API: 
Open your browser to `http://localhost:8000/docs` to access the interactive API documentation (Swagger UI).

---

### Docker Setup
Building the Docker Image
From the root directory of the project (where Dockerfile is located):

```Bash
docker build -t rag-chatbot:1.0.0 .
```
Replace 1.0.0 with your desired image tag.

### Running the Docker Container
To run the container locally, providing your environment variables from the `.env` file:

```Bash
docker run -d -p 8000:8000 --env-file ./.env rag-chatbot:1.0.0
```
`-d`: Runs the container in detached mode (in the background).
`-p 8000:8000`: Maps port `8000` on your host machine to port `8000` inside the container.
`--env-file ./.env`: Passes your environment variables from the `.env` file into the container.
Access the API at `http://localhost:8000/docs.`

---

## Kubernetes Deployment with Helm
This section outlines how one might go about deploying the application on Kubernetes using Helm charts.

### Helm Chart Structure
The Helm chart could be located in the rag-chatbot-chart/ directory with the following structure:
```
rag-chatbot-chart/
├── Chart.yaml                  # Metadata about the chart
├── values.yaml                 # Default configuration values for the chart
├── templates/                  # Directory for Kubernetes manifest templates
│   ├── _helpers.tpl            # Reusable template snippets
│   ├── deployment.yaml         # Kubernetes Deployment manifest
│   ├── service.yaml            # Kubernetes Service manifest
│   └── secret.yaml             # Kubernetes Secret manifest (for sensitive ENV vars)
└── .helmignore                 # Files to ignore when packaging the chart
```
### Deployment Steps
1. Build and Push Docker Image:
Ensure your Docker image is built and pushed to a container registry accessible by your Kubernetes cluster (e.g., Docker Hub, Google Container Registry, AWS ECR).
```Bash

docker build -t your-dockerhub-username/rag-chatbot:1.0.0 .
docker push your-dockerhub-username/rag-chatbot:1.0.0
```
Important: Update `rag-chatbot-chart/values.yaml` with your image.repository and `image.tag` to match your pushed image.

2. Configure `values.yaml`:
Edit `rag-chatbot-chart/values.yaml` to set non-sensitive environment variables and resource requests/limits. For sensitive variables (API keys), you can either:

- Directly in `values.yaml` (for demonstration/testing, NOT recommended for production): Replace placeholder values under `secretEnv`. Helm will `base64` encode these when creating the `Secret`.
- Securely via `helm install --set-string` (recommended): Provide sensitive values during installation.
- Using a Secret Management Solution: In production, integrate with a dedicated Kubernetes Secret management system (e.g., HashiCorp Vault, External Secrets Operator).

3. Navigate to Chart Directory:

```Bash

cd rag-chatbot-chart
```
4. Install the Helm Chart:

```Bash

helm install rag-chatbot . \
  --set-string secretEnv.OPENAI_API_KEY="your_actual_openai_key" \
  --set-string secretEnv.PINECONE_API_KEY="your_actual_pinecone_key" \
  --set-string secretEnv.PINECONE_ENVIRONMENT="your_pinecone_env" \
  --set image.repository="your-dockerhub-username/rag-chatbot" \
  --set image.tag="1.0.0" # Ensure this matches your pushed image tag
```
This command will deploy your application to the current Kubernetes context.

5. Verify Deployment:
Check the status of your pods and services:

```Bash

kubectl get pods -l app.kubernetes.io/name=rag-chatbot
kubectl get svc -l app.kubernetes.io/name=rag-chatbot
```
6. Access the Application:
- If service.type in `values.yaml` is LoadBalancer (and your cluster supports it), `kubectl get svc` will show an external IP.
- If ClusterIP, you'll need to use `kubectl port-forward` for local access or set up an Ingress controller for external access.

```Bash

kubectl port-forward svc/rag-chatbot 8000:8000
```
Then access `http://localhost:8000/docs`.

---

## API Endpoints
The API provides the following endpoints:

- `POST /ingest`:

  - **Description**: Ingests documents from a list of provided URLs. Fetches locally downloaded content mapped to it if available, optionally chunks it, generates embeddings, and upserts them to Pinecone.
  - **Request Body**:
    ```JSON    
        {
        "urls": ["https://example.com/doc1", "https://example.com/doc2"],
        "chunk_documents": true
        }
    ```    
  - **Response**: `IngestResponse` indicating success/failure and lists of ingested/failed URLs.

- `POST /chat`:

  - **Description**: Accepts a user query, retrieves relevant context from Pinecone, and generates a conversational response using the configured LLM.
  - **Request Body**:
    ```JSON

    {
    "message": "What is the capital of France?"
    }
    ```
  - **Response**: `ChatResponse` containing the generated response and source document snippets.

---

## Error Handling & Best Practices
- **Logging**: Comprehensive logging is implemented using Python's logging module throughout `rag_pipeline.py` and `app.py` to provide visibility into application flow, document processing, and errors.
- **API Error Handling**: FastAPI's `HTTPException` is used to return appropriate HTTP status codes (e.g., `400 Bad Request`, `500 Internal Server Error`, `503 Service Unavailable`) and detailed error messages to the client.
- **Robust Document Fetching**: Uses `httpx` with timeout and `raise_for_status()` to handle network issues and bad HTTP responses during URL fetching.
- **Pinecone Initialization Check**: The `rag_pipeline` explicitly checks for Pinecone initialization status and handles cases where API keys or environment variables are missing, preventing operations on an uninitialized client.
- **Graceful LLM/Embedding Failures**: try-except blocks around LiteLLM calls catch potential issues during embedding or completion generation, returning informative error messages.
- **Demonstration of Resource Management (Kubernetes)**: The Helm chart includes resources limits and requests in `deployment.yaml` to show how one might attempt to ensure pods consume appropriate CPU and memory, helping to prevent resource exhaustion and improve cluster stability.
- **Demonstration of Liveness and Readiness Probes (Kubernetes)**: Configured in `deployment.yaml` to illustrate how Kubernetes can monitor the health of the application pods, enabling automatic restarts of unhealthy pods and preventing traffic from being sent to unready instances.
- **Demonstration of Secrets Management**: Sensitive API keys are managed as Kubernetes Secrets via the Helm chart, illustrating a standard and more secure practice than hardcoding them directly into images or `ConfigMaps`.

---

## Architectural Decisions & Trade-offs
1. **RAG Architecture Choice**:
- **Decision**: Implemented a standard RAG pattern (retrieve-then-generate).
- **Trade-offs**:
  - **Pros**: Reduces LLM hallucinations, grounds responses in factual data, allows dynamic updating of knowledge base without retraining the LLM, cost-effective as it reuses smaller LLMs.
  - **Cons**: Requires managing an external vector database (Pinecone), latency introduced by retrieval step, quality of response heavily depends on retrieval relevance and chunking strategy.

2. **LiteLLM for LLM/Embedding Abstraction**:
- **Decision**: Used LiteLLM to abstract away different LLM and embedding model providers.
- **Trade-offs**:
  - **Pros**: Future-proof (easy to switch providers/models), simplifies API calls, reduces vendor lock-in.
  - **Cons**: Adds an extra dependency layer, potential for slight overhead compared to direct API calls (negligible for most use cases).

3. **Pinecone as Vector Database**:
- **Decision**: Chosen for its managed service, scalability, and ease of integration.
- **Trade-offs**:
  - **Pros**: No self-hosting overhead, highly scalable, optimized for vector search.
  - **Cons**: SaaS dependency, potential cost implications for high usage, requires network access.

4. **FastAPI for API Development**:
- **Decision**: Selected for its modern Python features, asynchronous support, and automatic OpenAPI documentation.
- **Trade-offs:**
  - **Pros**: High performance (thanks to Starlette/Uvicorn), excellent developer experience, strong type hints with Pydantic.
  - **Cons**: Requires understanding of asynchronous programming (async/await).

5. **Document Chunking Strategy**:
- **Decision**: Used `langchain-text-splitters` with `RecursiveCharacterTextSplitter`.
- **Trade-offs**:
  - **Pros**: Flexible and robust for various text types, handles overlaps to preserve context.
  - **Cons**: Optimal `chunk_size` and `chunk_overlap` are often heuristic and might require tuning for different document types to maximize retrieval relevance. Not all information might fit into a single chunk, potentially splitting critical context.

6. **Containerization with Docker**:
- **Decision**: Packaged the application into a Docker image.
- **Trade-offs**:
  - **Pros**: Ensures consistent environments across development, testing, and production; simplifies dependency management; provides isolation.
  - **Cons**: Adds a layer of abstraction and requires Docker knowledge.

7. **Demonstration of Kubernetes Deployment with Helm**:
- **Decision**: Provided Helm charts for Kubernetes deployment as a demonstration of potential deployment.
- **Trade-offs**:
  - **Pros**: Standardized way to package and deploy Kubernetes applications, enables versioning and easy upgrades/rollbacks, externalizes configuration, facilitates CI/CD.
  - **Cons**: Adds complexity to the deployment process compared to simple Docker runs; requires Kubernetes and Helm knowledge. This particular setup is illustrative and might require further hardening for robust production environments.

This comprehensive setup allows for a flexible, scalable, and maintainable RAG chatbot solution.