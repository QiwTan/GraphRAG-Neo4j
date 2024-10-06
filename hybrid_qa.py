# interactive_qa.py

import os
from typing import List, Tuple

import config

# Neo4j and GraphDatabase
from neo4j import GraphDatabase

# LangChain imports
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from langchain.runnables import RunnableParallel, RunnablePassthrough
from langchain.schema import HumanMessage, AIMessage
from langchain.runnables import RunnableBranch, RunnableLambda
from langchain.embeddings import OllamaEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.vectorstores import Neo4jVector
from langchain.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOllama

def initialize_chain():
    # Connect to Neo4j
    os.environ["NEO4J_URI"] = config.NEO4J_URI
    os.environ["NEO4J_USERNAME"] = config.NEO4J_USERNAME
    os.environ["NEO4J_PASSWORD"] = config.NEO4J_PASSWORD

    # Initialize models and embeddings
    llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    llm2 = ChatOllama(model=config.LLM_MODEL)
    embedding = OllamaEmbeddings(model=config.EMBEDDING_MODEL, temperature=config.LLM_TEMPERATURE)

    # Initialize Neo4j graph
    graph = Neo4jGraph()

    # Initialize vector index
    vector_index = Neo4jVector.from_existing_graph(
        embedding,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    # Create full-text index if not exists
    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
    )

    # Extract entities from text
    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
            ...,
            description="All the entities that appear in the text",
        )

    # Define prompt template for entity extraction
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting entities from the text.",
            ),
            (
                "human",
                "Use the json format to extract information from the following "
                "input: {question}",
            ),
        ]
    )

    # Define entity extraction chain
    entity_chain = prompt | llm.with_structured_output(Entities)

    def generate_full_text_query(input: str) -> str:
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    # Full-text index query
    def structured_retriever(question: str) -> str:
        result = ""
        entities = entity_chain.invoke({"question": question})
        print("Extracted Entities:", entities.names)
        for entity in entities.names:
            response = graph.query(
                """
                CALL db.index.fulltext.queryNodes('entity', $query)
                YIELD node, score
                CALL (node) {
                    WITH node
                    MATCH (node)-[r]->(neighbor)
                    WHERE NOT node:Document AND NOT neighbor:Document
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    MATCH (node)<-[r]-(neighbor)
                    WHERE NOT node:Document AND NOT neighbor:Document
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 200
                """,
                {"query": generate_full_text_query(entity)},
            )
            if response:
                result += "\n".join([el['output'] for el in response])
            else:
                result += f"No relationships found for entity: {entity}\n"
        return result

    def retriever(question: str):
        print(f"Search query: {question}")
        structured_data = structured_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question, k=5)]
        final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ".join(unstructured_data)}
        """
        print(final_data)
        
        return final_data

    # Condense chat history and follow-up question into a standalone question
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x: x["question"]),
    )

    template = """
    You are an expert research assistant. Based on the provided context, answer the following question in detail, including a summary of relevant sections such as methods, results, and conclusions when applicable.

    Context:
    {context}

    Question: {question}

    Your answer should:
    - Provide a comprehensive explanation, including key points from the document.
    - If applicable, include a summary of methodology, results, and conclusions.
    - Ensure the response is clear and well-organized for a researcher to easily understand.

    Answer in detail:
    """

    prompt_template = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": _search_query | retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt_template
        | llm2
        | StrOutputParser()
    )

    return chain

def main():
    # Initialize chain
    chain = initialize_chain()
    print("System initialized. You can start asking questions. Type 'exit' or 'quit' to exit.")

    while True:
        try:
            # Accept user input
            user_question = input("Enter your question: ").strip()
            
            if user_question.lower() in ['exit', 'quit']:
                print("Exiting the program.")
                break

            if not user_question:
                print("Question cannot be empty. Please try again.")
                continue

            # Invoke chain for query
            response = chain.invoke({"question": user_question})
            print(f"Answer: {response}")

        except KeyboardInterrupt:
            print("\nDetected interruption, exiting the program.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()