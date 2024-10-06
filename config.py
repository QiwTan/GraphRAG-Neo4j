# config.py

# Neo4j credentials
NEO4J_URI = "neo4j://192.168.0.112:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_password_here"

# Input and output folders
INPUT_PDF_FOLDER = "/Users/tanqiwen/Documents/GraphRAG-Neo4j"
OUTPUT_JSON_FOLDER = "/Users/tanqiwen/Documents/GraphRAG-Neo4j"

# Text splitting parameters
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 400

# Ollama models
LLM_MODEL = "llama3.1:latest"
LLM_TEMPERATURE = 0
EMBEDDING_MODEL = "nomic-embed-text:latest"