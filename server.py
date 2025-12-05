"""
title: Legal Document Hybrid RAG Pipeline
authors: your-name
author_url: https://github.com/your-username
funding_url: https://github.com/open-webui
version: 2.0.0
required_open_webui_version: 0.3.17
license: MIT
"""

import os
import json
import mysql.connector
from typing import List, Union, Generator, Iterator, Dict
from pydantic import BaseModel, Field


class Pipe:
    class Valves(BaseModel):
        MYSQL_HOST: str = Field(default="localhost")
        MYSQL_PORT: int = Field(default=3306)
        MYSQL_USER: str = Field(default="root")
        MYSQL_PASSWORD: str = Field(default="admin")
        MYSQL_DATABASE: str = Field(default="flask_application")
        DOCS_BASE_PATH: str = Field(
            default="C:/legal_documents/",
            description="Base path where legal documents are stored",
        )
        CHROMA_PATH: str = Field(
            default="C:/legal_documents/chroma_db",
            description="Path to ChromaDB vector database",
        )
        MAX_DOCUMENTS: int = Field(
            default=10, description="Maximum number of documents to retrieve"
        )
        KEYWORD_CANDIDATES: int = Field(
            default=20, description="Number of candidates from keyword search"
        )
        VECTOR_CANDIDATES: int = Field(
            default=20, description="Number of candidates from vector search"
        )
        USE_HYBRID_SEARCH: bool = Field(
            default=True, description="Use hybrid search (keyword + vector)"
        )
        USE_LLM_EXTRACTION: bool = Field(
            default=True, description="Use LLM for intelligent keyword extraction"
        )
        OLLAMA_BASE_URL: str = Field(
            default="http://localhost:11434", description="Ollama API base URL"
        )
        OLLAMA_MODEL: str = Field(
            default="deepseek-v3.1:671b-cloud",
            description="Ollama model to use for keyword extraction",
        )
        RESPONSE_MODEL: str = Field(
            default="deepseek-v3.1:671b-cloud",
            description="Model to use for generating final answers",
        )
        EMBEDDING_MODEL: str = Field(
            default="all-MiniLM-L6-v2",
            description="Sentence transformer model for embeddings",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "legal_rag"
        self.name = "legal_rag/"
        self.valves = self.Valves(
            **{
                "MYSQL_HOST": os.getenv("MYSQL_HOST", "localhost"),
                "MYSQL_PORT": int(os.getenv("MYSQL_PORT", "3306")),
                "MYSQL_USER": os.getenv("MYSQL_USER", ""),
                "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD", ""),
                "MYSQL_DATABASE": os.getenv("MYSQL_DATABASE", ""),
            }
        )
        self.chroma_client = None
        self.chroma_collection = None

    def get_models(self):
        return [
            {"id": "legal-rag", "name": "legal-rag"},
            {"id": "legal-search", "name": "legal-search"},
            {"id": "legal-hybrid", "name": "legal-hybrid"},
        ]

    def pipes(self) -> List[dict]:
        return self.get_models()

    def get_db_connection(self):
        """Create MySQL database connection"""
        try:
            connection = mysql.connector.connect(
                host=self.valves.MYSQL_HOST,
                port=self.valves.MYSQL_PORT,
                user=self.valves.MYSQL_USER,
                password=self.valves.MYSQL_PASSWORD,
                database=self.valves.MYSQL_DATABASE,
            )
            return connection
        except mysql.connector.Error as e:
            raise Exception(f"Failed to connect to MySQL: {e}")

    def get_chroma_collection(self):
        """Get or create ChromaDB collection"""
        if self.chroma_collection is not None:
            return self.chroma_collection

        try:
            import chromadb
            from chromadb.config import Settings

            # Initialize ChromaDB client
            if self.chroma_client is None:
                self.chroma_client = chromadb.PersistentClient(
                    path=self.valves.CHROMA_PATH,
                    settings=Settings(anonymized_telemetry=False),
                )

            # Get or create collection
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name="legal_documents"
                )
            except:
                # Create collection if it doesn't exist
                self.chroma_collection = self.chroma_client.create_collection(
                    name="legal_documents",
                    metadata={"hnsw:space": "cosine"},
                )

            return self.chroma_collection

        except Exception as e:
            print(f"ChromaDB initialization error: {e}")
            return None

    def extract_keywords(self, query: str, use_llm: bool = True) -> List[str]:
        """Extract keywords from user query"""

        if use_llm:
            try:
                import requests

                extraction_prompt = f"""You are a keyword extraction system. Extract ONLY the important keywords from this legal query.

Query: "{query}"

Output ONLY comma-separated keywords. Do not write anything else. No explanations, no sentences, just keywords.

Example output: defendant, liability, damages, plaintiff
Your keywords:"""

                response = requests.post(
                    f"{self.valves.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": self.valves.OLLAMA_MODEL,
                        "prompt": extraction_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 50,
                            "stop": ["\n", ".", "!"],
                        },
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    result = response.json()
                    keywords_text = result.get("response", "").strip()

                    prefixes_to_remove = [
                        "here is the extracted list of important keywords and legal terms:",
                        "here are the keywords:",
                        "keywords:",
                        "here is:",
                        "the keywords are:",
                    ]

                    keywords_text_lower = keywords_text.lower()
                    for prefix in prefixes_to_remove:
                        if keywords_text_lower.startswith(prefix):
                            keywords_text = keywords_text[len(prefix) :].strip()
                            break

                    keywords_text = keywords_text.split("\n")[0].strip()
                    keywords_text = keywords_text.rstrip(".!?;:")
                    keywords = [
                        k.strip().lower() for k in keywords_text.split(",") if k.strip()
                    ]
                    keywords = [
                        k for k in keywords if len(k.split()) <= 3 and len(k) < 30
                    ]

                    if keywords and len(keywords) >= 2:
                        print(f"LLM extracted keywords: {keywords}")
                        return keywords[:10]

            except Exception as e:
                print(f"LLM extraction failed: {e}, falling back to simple method")

        # Fallback: Simple keyword extraction
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "what",
            "when",
            "where",
            "who",
            "how",
            "can",
            "could",
            "would",
            "should",
            "has",
            "have",
            "had",
            "been",
            "being",
            "this",
            "that",
            "these",
            "those",
        }

        words = query.lower().split()
        keywords = [
            w.strip(".,!?;:")
            for w in words
            if w.lower() not in stop_words and len(w) > 2
        ]

        return keywords[:10]

    def search_documents_keyword(
        self, keywords: List[str], limit: int = 20
    ) -> List[dict]:
        """Keyword-based search using MySQL"""
        connection = None
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor(dictionary=True)

            keyword_conditions = []
            params = []

            for keyword in keywords:
                keyword_conditions.append("(keywords LIKE %s OR document_name LIKE %s)")
                search_term = f"%{keyword}%"
                params.extend([search_term, search_term])

            query = f"""
                SELECT id, document_name, file_path, page_range, section, keywords
                FROM legal_document
                WHERE {' OR '.join(keyword_conditions)}
                LIMIT %s
            """
            params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            return results

        except Exception as e:
            print(f"Keyword search error: {e}")
            return []
        finally:
            if connection and connection.is_connected():
                connection.close()

    def search_documents_vector(self, query: str, limit: int = 20) -> List[dict]:
        """Vector-based semantic search using ChromaDB"""
        try:
            collection = self.get_chroma_collection()
            if collection is None:
                print("ChromaDB not available, skipping vector search")
                return []

            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=limit,
            )

            # Convert ChromaDB results to our format
            documents = []
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    metadata = (
                        results["metadatas"][0][i] if results.get("metadatas") else {}
                    )
                    distance = (
                        results["distances"][0][i] if results.get("distances") else 0
                    )

                    documents.append(
                        {
                            "id": metadata.get("db_id", doc_id),
                            "document_name": metadata.get("document_name", ""),
                            "file_path": metadata.get("file_path", ""),
                            "page_range": metadata.get("page_range", "N/A"),
                            "section": metadata.get("section", "N/A"),
                            "keywords": metadata.get("keywords", ""),
                            "vector_score": 1
                            - distance,  # Convert distance to similarity
                        }
                    )

            return documents

        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def reciprocal_rank_fusion(
        self, keyword_results: List[dict], vector_results: List[dict], k: int = 60
    ) -> List[dict]:
        """
        Combine keyword and vector search results using Reciprocal Rank Fusion (RRF)

        RRF formula: score = sum(1 / (k + rank))
        where k is a constant (typically 60) and rank is the position in the result list
        """
        scores = {}
        doc_data = {}

        # Score keyword results
        for rank, doc in enumerate(keyword_results, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank))
            doc_data[doc_id] = doc

        # Score vector results
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank))
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

        # Sort by RRF score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return documents with scores
        results = []
        for doc_id, score in sorted_docs:
            doc = doc_data[doc_id].copy()
            doc["rrf_score"] = score
            results.append(doc)

        return results

    def hybrid_search(self, query: str, keywords: List[str]) -> List[dict]:
        """
        Perform hybrid search: keyword + vector with RRF fusion
        """
        # Get candidates from both searches
        keyword_results = self.search_documents_keyword(
            keywords, limit=self.valves.KEYWORD_CANDIDATES
        )
        vector_results = self.search_documents_vector(
            query, limit=self.valves.VECTOR_CANDIDATES
        )

        print(f"Keyword search found: {len(keyword_results)} documents")
        print(f"Vector search found: {len(vector_results)} documents")

        # Combine using RRF
        fused_results = self.reciprocal_rank_fusion(keyword_results, vector_results)

        # Return top MAX_DOCUMENTS
        return fused_results[: self.valves.MAX_DOCUMENTS]

    def read_document_content(self, file_path: str) -> str:
        """Read document content from file system"""
        try:
            full_path = os.path.join(self.valves.DOCS_BASE_PATH, file_path)

            if not os.path.exists(full_path):
                return f"[Document not found: {full_path}]"

            if full_path.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif full_path.endswith(".pdf"):
                return "[PDF reading not yet implemented]"
            else:
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()

        except Exception as e:
            return f"[Error reading document: {e}]"

    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with legal system prompt"""
        try:
            import requests

            system_prompt = """You are the solicitors for the 5th Defendant.
Rules:
- Always provide citations with both exact quoted lines AND any available location markers (page/section if given).
- Give answers using provided context.
- Words/Expressions may have different and many meanings depending on different agreements, schedules and documents.
- You may come up with your own assumptions and formulas based on definitions in context if the formulas are not provided directly in context.
- Show all your workings and formulas during the process of calculations.
- Combine all chunks because they have relevant contents.
- Search the page number based on the header. For example: ---PAGE 24---
- Include the page or section reference from the context.

Output format:
(a) Answer
(b) Supporting Quotes (with page/section if present)"""

            full_prompt = f"""{system_prompt}

Text:
<<<
{context}
>>>

Question: {query}"""

            response = requests.post(
                f"{self.valves.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.valves.RESPONSE_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2000,
                    },
                },
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                return answer
            else:
                return f"Error generating answer: HTTP {response.status_code}"

        except Exception as e:
            return f"Error generating answer: {e}"

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        try:
            model = body["model"][body["model"].find(".") + 1 :]
            messages = body.get("messages", [])

            if not messages:
                return "No query provided."

            user_query = ""
            if isinstance(messages[-1].get("content"), str):
                user_query = messages[-1]["content"]
            elif isinstance(messages[-1].get("content"), list):
                for item in messages[-1]["content"]:
                    if item.get("type") == "text":
                        user_query = item["text"]
                        break

            if model == "legal-search":
                return self.search_only(user_query)
            elif model == "legal-rag":
                return self.rag_pipeline(user_query, body, messages, use_hybrid=False)
            elif model == "legal-hybrid":
                return self.rag_pipeline(user_query, body, messages, use_hybrid=True)
            else:
                return f"Unknown model: {model}"

        except Exception as e:
            return f"Error: {e}"

    def search_only(self, query: str) -> str:
        """Search and display results without RAG"""
        try:
            keywords = self.extract_keywords(
                query, use_llm=self.valves.USE_LLM_EXTRACTION
            )

            response = f"🔍 **Search Results for:** {query}\n\n"
            response += f"**Extracted Keywords:** {', '.join(keywords)}\n\n"

            if self.valves.USE_HYBRID_SEARCH:
                documents = self.hybrid_search(query, keywords)
                response += "**Search Method:** Hybrid (Keyword + Vector with RRF)\n\n"
            else:
                documents = self.search_documents_keyword(
                    keywords, self.valves.MAX_DOCUMENTS
                )
                response += "**Search Method:** Keyword Only\n\n"

            if not documents:
                response += "❌ No documents found matching your query.\n"
                return response

            response += f"✅ **Found {len(documents)} relevant documents:**\n\n"

            for i, doc in enumerate(documents, 1):
                response += f"**{i}. {doc['document_name']}**\n"
                response += f"   📄 Pages: {doc.get('page_range', 'N/A')}\n"
                response += f"   📁 Section: {doc.get('section', 'N/A')}\n"
                response += f"   📍 Path: {doc['file_path']}\n"
                if doc.get("rrf_score"):
                    response += f"   ⭐ Relevance Score: {doc['rrf_score']:.4f}\n"
                response += f"   🔖 Keywords: {doc.get('keywords', 'N/A')}\n\n"

            return response

        except Exception as e:
            return f"Error in search: {e}"

    def rag_pipeline(
        self, query: str, body: dict, messages: list, use_hybrid: bool = True
    ) -> str:
        """Full RAG pipeline with optional hybrid search"""
        try:
            keywords = self.extract_keywords(
                query, use_llm=self.valves.USE_LLM_EXTRACTION
            )

            # Use hybrid or keyword-only search
            if use_hybrid and self.valves.USE_HYBRID_SEARCH:
                documents = self.hybrid_search(query, keywords)
                search_method = "Hybrid (Keyword + Vector)"
            else:
                documents = self.search_documents_keyword(
                    keywords, self.valves.MAX_DOCUMENTS
                )
                search_method = "Keyword Only"

            if not documents:
                return f"❌ No relevant legal documents found.\n\n**Query:** {query}\n**Keywords:** {', '.join(keywords)}\n**Search Method:** {search_method}"

            # Retrieve document content
            context_parts = []
            doc_references = []

            for i, doc in enumerate(documents, 1):
                content = self.read_document_content(doc["file_path"])

                doc_header = f"=== DOCUMENT {i}: {doc['document_name']} ===\n"
                doc_header += f"Pages: {doc.get('page_range', 'N/A')}\n"
                doc_header += f"Section: {doc.get('section', 'N/A')}\n"
                if doc.get("rrf_score"):
                    doc_header += f"Relevance Score: {doc['rrf_score']:.4f}\n"
                doc_header += f"---\n"

                doc_references.append(
                    f"[{i}] {doc['document_name']} (Pages {doc.get('page_range', 'N/A')})"
                )

                context_parts.append(f"{doc_header}{content}")

            full_context = "\n\n".join(context_parts)

            # Generate answer
            answer = self.generate_answer(query, full_context)

            # Format response
            response = "📚 **Retrieved Documents:**\n"
            for ref in doc_references:
                response += f"• {ref}\n"
            response += f"\n**Search Method:** {search_method}\n"
            response += f"**Keywords used:** {', '.join(keywords)}\n"
            response += "\n" + "=" * 50 + "\n\n"
            response += f"**Question:** {query}\n\n"
            response += f"{answer}\n"

            return response

        except Exception as e:
            return f"Error in RAG pipeline: {e}"
