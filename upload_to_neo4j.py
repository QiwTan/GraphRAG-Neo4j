# upload_to_neo4j.py

import os
import json
from tqdm import tqdm
from typing import List

import config

# Neo4j and GraphDatabase
from neo4j import GraphDatabase

# LangChain imports
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OllamaEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.vectorstores import Neo4jVector
from langchain.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_experimental.graph_transformers import LLMGraphTransformer
import ollama_functions as OllamaFunctions
from langchain.chat_models import ChatOllama

# Updated ChatPromptTemplate with corrected placeholder
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    Persona:
    You are an expert research assistant specializing in scientific literature analysis for STEM fields, focused on extracting key information from academic papers.

    Goal:
    Given a document, extract all entities and relationships from the following sections:
    - **Abstract**: High-level summary, goals, and key results.
    - **Introduction**: Research problem, hypotheses, and objectives.
    - **Methodology**: Main methods, tools, datasets, and technologies.
    - **Results**: Key findings and outcomes.
    - **Conclusion**: Overall findings and implications.

    Allowed Nodes:
    - Paper, Author, Organization, Methodology, Dataset, Result, Evaluation_Metric, Research_Direction, Hypothesis, Tool, Theory, Conclusion, Limitations.

    Allowed Relationships:
    - Authored_By, Affiliated_With, Uses_Method, Reports_Result, Evaluated_By, Uses_Dataset, Related_To, Tests_Hypothesis, Utilizes_Tool, Builds_On_Theory, Draws_Conclusion, Mentions_Limitations.

    Steps:
    1. Identify entities with:
       - `entity_name`: Capitalized name of the entity.
       - `entity_type`: One of the allowed nodes.
       
    2. Identify related pairs (source_entity, target_entity) with:
       - `source_entity`: Name of the source entity.
       - `target_entity`: Name of the target entity.

    3. Ignore mathematical formulas or symbols.

    4. Return output in clear, structured English by sections.

    5. Translate non-English descriptions while keeping everything else intact.
    """),
    ("human", 
     "Extract the following information from the literature chunk titled '{main_title}': \n\ntext: {page_content}")
])

def initialize():
    # Connect to Neo4j
    os.environ["NEO4J_URI"] = config.NEO4J_URI
    os.environ["NEO4J_USERNAME"] = config.NEO4J_USERNAME
    os.environ["NEO4J_PASSWORD"] = config.NEO4J_PASSWORD

    # Initialize LLM and embeddings
    llm = OllamaFunctions(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE, prompt=chat_prompt)

    # Initialize Neo4j graph
    graph = Neo4jGraph()

    # Read chunked documents from specified directory
    json_folder_path = config.OUTPUT_JSON_FOLDER
    chunked_files = [f for f in os.listdir(json_folder_path) if f.endswith("_chunked.json")]

    for file_name in tqdm(chunked_files, desc="Processing Files"):
        json_file_path = os.path.join(json_folder_path, file_name)
        with open(json_file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        raw_documents = []
        # Convert each chunk to Document object with necessary metadata
        for chunk in chunks_data:
            content = chunk.get("chunk_content", "")
            metadata = {
                "main_title": chunk.get("main_title", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "chunk_length": len(content),
                "document_content": content
            }
            doc = Document(page_content=content, metadata=metadata)
            raw_documents.append(doc)

        documents = raw_documents

        # Convert to graph document
        llm_transformer_filtered = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=[
                "Paper", "Author", "Organization", "Methodology", "Dataset", "Result",
                "Evaluation_Metric", "Research_Direction", "Hypothesis", "Tool", "Theory",
                "Conclusion", "Limitations"
            ],
            allowed_relationships=[
                "Authored_By", "Affiliated_With", "Uses_Method", "Reports_Result", "Evaluated_By",
                "Uses_Dataset", "Related_To", "Tests_Hypothesis", "Utilizes_Tool", "Builds_On_Theory",
                "Draws_Conclusion", "Mentions_Limitations"
            ]
        )

        # Create full-text index if not exists
        graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
        )

        total_documents = len(documents)
        for i, doc in tqdm(enumerate(documents), total=total_documents, desc="Processing Documents"):
            # Format prompt
            formatted_prompt = chat_prompt.format(
                page_content=doc.page_content, 
                main_title=doc.metadata.get("main_title", "")
            )

            # Process current document
            graph_document = llm_transformer_filtered.convert_to_graph_documents([doc])[0]

            # Add to connected Neo4j graph
            graph.add_graph_documents(
                [graph_document], 
                baseEntityLabel=True, 
                include_source=True
            )

    print("Knowledge graph upload completed and vector index built.")

if __name__ == "__main__":
    initialize()