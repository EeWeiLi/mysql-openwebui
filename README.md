# Legal Document Hybrid RAG Pipeline

This project provides a Legal Document Hybrid Retrieval-Augmented Generation (RAG) pipeline to connect Open WebUI with MySQL for managing and querying legal documents. The pipeline combines keyword-based search and semantic vector search using ChromaDB, enhanced by a Legal Language Model (LLM) for intelligent keyword extraction and document-based question answering.

The Legal Document Hybrid RAG Pipeline is designed to:

1. ## Search legal documents using a combination of keyword-based search and semantic vector search.

2. ## Extract important keywords from queries using a Legal Language Model (LLM).

3. ## Retrieve relevant documents based on the keywords and vectors.

4. ## Generate answers using context from the retrieved documents with an LLM-based response generator.

The server integrates Open WebUI with MySQL for storing legal documents and ChromaDB for vector-based semantic search. This system is customizable to fine-tune the search methods, keyword extraction, and response generation for legal domains.

# Installation

## Dependencies

To run the project, first install the necessary dependencies using pip:

pip install -r requirements.txt

Make sure you have MySQL and ChromaDB set up before running the application.

Configuration
MySQL Settings

The MySQL connection settings are configured in the Valves class inside the code:

MYSQL_HOST: The host of the MySQL server (localhost by default).

MYSQL_PORT: The port for MySQL (default: 3306).

MYSQL_USER: MySQL user for connecting.

MYSQL_PASSWORD: The password for MySQL user.

MYSQL_DATABASE: The MySQL database containing legal documents.

You can set these as environment variables or modify the default values directly.

ChromaDB Settings

CHROMA_PATH: Path to the ChromaDB vector database (default: C:/legal_documents/chroma_db).

Make sure that ChromaDB is installed and that the path is set correctly to your database storage.

LLM Settings

OLLAMA_BASE_URL: The URL to connect to the Ollama API.

OLLAMA_MODEL: The model used for keyword extraction (deepseek-v3.1:671b-cloud by default).

RESPONSE_MODEL: The model used for generating answers (deepseek-v3.1:671b-cloud by default).

These values allow you to switch between different environments or LLM models as necessary.

Usage
Search Only

To perform a search without using the full RAG pipeline, the search_only function allows you to perform keyword-based search on the query:

query = "What is the definition of negligence in law?"
response = pipe.search_only(query)
print(response)


This method will display search results using keyword-based retrieval from MySQL or vector search from ChromaDB, based on the configuration.

Full RAG Pipeline

The RAG pipeline involves extracting keywords from the query, performing keyword-based and vector-based search, and generating answers based on the retrieved documents. Here's how to use it:

query = "What are the legal requirements for a contract in Malaysia?"
response = pipe.rag_pipeline(query, body, messages)
print(response)


The pipe.rag_pipeline method retrieves documents using both keyword and vector search, and generates an answer using the context of these documents.

License

This project is licensed under the MIT License.

For more details, check the LICENSE file
.

Example Output for rag_pipeline
📚 **Retrieved Documents:**
• [1] Contract Law in Malaysia (Pages 1-5)
• [2] Business Contracts in Malaysia (Pages 6-10)

**Search Method:** Hybrid (Keyword + Vector)
**Keywords used:** contract, law, Malaysia

==================================================
**Question:** What are the legal requirements for a contract in Malaysia?

Answer:
(a) A contract in Malaysia must contain:
    1. An offer and acceptance.
    2. Consideration.
    3. Intention to create legal relations.
    4. Legal capacity of parties involved.
(b) Supporting Quotes:
    - "An offer and acceptance" (Page 3)
    - "Consideration" (Page 5)

Open WebUI Integration

The Open WebUI server can be used to connect directly with MySQL for storing and querying legal documents. Here’s how it functions:

Connecting Open WebUI with MySQL: The server retrieves documents stored in MySQL, performs searches using the provided models, and generates answers.

Embedding this system: This can be integrated into the Open WebUI function tab as part of a custom legal document retrieval pipeline.

To use Open WebUI with MySQL, configure the MySQL connection as mentioned above and use the functions in the Pipe class for document retrieval, search, and response generation.
