# GraphRAG-Neo4j

This repository provides a pipeline for processing, uploading, and querying scientific literature in PDF format. The pipeline extracts entities and relationships from the literature and stores them in a Neo4j knowledge graph for retrieval and analysis.

## Features

1. **PDF Preprocessing**:
    - Cleans and splits PDF files into text chunks.
    - Extracts main titles and section headings.
    - Removes content after references.
    - Saves processed chunks as JSON files.

2. **Data Upload to Neo4j**:
    - Uses an LLM (Ollama) to extract entities and relationships from the literature chunks.
    - Uploads entities and relationships to a Neo4j knowledge graph.
    - Builds a vector index for efficient retrieval.

3. **Interactive Question Answering**:
    - Provides an interactive interface for querying the knowledge graph.
    - Supports hybrid search using full-text and vector-based retrieval.
    - Extracts structured and unstructured data from the knowledge graph.

## Prerequisites

- **Python 3.x** installed
- **Neo4j** running on your local or remote server
- **Required Python libraries**:
    - `neo4j`
    - `langchain`
    - `tqdm`
    - `PyPDF2`
    - `ollama_functions`
    - `pydantic`
    - `json`

You can install the necessary Python libraries with the following command:

```bash
pip install neo4j langchain langchain-community langchain_experimental tqdm PyPDF2 pydantic subprocess sys json
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/QiwTan/GraphRAG-Neo4j.git
    ```
2. Navigate to the repository directory:
    ```bash
    cd <repository_directory>
    ```

3. Create a configuration file by editing `config.py`:
    ```python
    # config.py
    NEO4J_URI = "neo4j://your_neo4j_host:7687"
    NEO4J_USERNAME = "your_neo4j_username"
    NEO4J_PASSWORD = "your_neo4j_password"

    INPUT_PDF_FOLDER = "PDF_data/"
    OUTPUT_JSON_FOLDER = "PDF_json/chunked/"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    LLM_MODEL = "llama3.1:latest"
    LLM_TEMPERATURE = 0
    EMBEDDING_MODEL = "nomic-embed-text:latest"
    ```

4. Ensure Neo4j is running and accessible using the credentials provided in the `config.py`.

## Usage

### 1. Preprocess PDFs

To preprocess PDFs and generate JSON chunks, run the following command:

```bash
python preprocess_pdfs.py
```

This script reads all PDF files from the `PDF_data/` directory, cleans and splits them, and saves the output as JSON chunks in the `PDF_json/chunked/` directory.

### 2. Upload Data to Neo4j

To upload the processed data to Neo4j, run the following command:

```bash
python upload_to_neo4j.py
```

This script extracts entities and relationships from the JSON chunks using the LLM and uploads them to Neo4j, creating a knowledge graph.

### 3. Hybrid QA

To start the Hybrid question-answering interface, run:

```bash
python hybrid_qa.py
```

This will allow you to ask questions related to the uploaded literature and get detailed responses based on the knowledge graph.

### 4. Run All Steps

To run the entire workflow (preprocessing, upload, and QA), execute the following command:

```bash
python main.py
```

Follow the on-screen prompts to select the operation you want to perform.

## File Structure

```
├── config.py              # Configuration file with paths and Neo4j credentials
├── main.py                # Main script to run all steps
├── preprocess_pdfs.py     # Preprocesses PDF files into JSON chunks
├── upload_to_neo4j.py     # Uploads JSON chunks to Neo4j
├── interactive_qa.py      # Interactive question-answering script
├── PDF_data/              # Folder containing input PDF files
└── PDF_json/chunked/      # Folder containing output JSON chunks
```