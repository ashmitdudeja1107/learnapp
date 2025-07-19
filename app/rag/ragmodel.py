import chromadb
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple, Optional
import uuid
import io
import re
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import shutil

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, collection_name: str = "ai_tutor_docs", embedding_method: str = "tfidf", reset_collection: bool = False):
        """
        Initialize RAG system with different embedding options:
        - 'tfidf': Use TF-IDF vectorization (no external dependencies)
        - 'openai': Use OpenAI embeddings (requires API key)
        - 'custom': Use custom embedding function
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_method: Method for generating embeddings
            reset_collection: If True, delete and recreate the collection
        """
        # Initialize ChromaDB with new configuration
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Set embedding method
        self.embedding_method = embedding_method
        self.collection_name = collection_name
        
        # Initialize embedding components based on method
        if embedding_method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=1000,  # Reduced for more consistent dimensions
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.embedding_function = self._get_tfidf_embedding
            self.expected_dimension = 1000
        elif embedding_method == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            self.openai_client = openai.OpenAI()
            self.embedding_function = self._get_openai_embedding
            self.expected_dimension = 1536
        elif embedding_method == "custom":
            self.embedding_function = self._get_custom_embedding
            self.expected_dimension = 384
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")
        
        # Storage for documents (needed for TF-IDF)
        self.document_store = []
        self.fitted_vectorizer = False
        
        # Handle collection creation/loading with dimension compatibility
        self.collection = self._initialize_collection(reset_collection)
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Print initial state
        doc_count = self.collection.count()
        print(f"Collection initialized with {doc_count} documents using {embedding_method} embeddings")
        print(f"Expected embedding dimension: {self.expected_dimension}")
    
    def _initialize_collection(self, reset_collection: bool):
        """Initialize collection with proper dimension handling"""
        try:
            # If reset is requested, delete existing collection
            if reset_collection:
                try:
                    self.client.delete_collection(self.collection_name)
                    print(f"Deleted existing collection: {self.collection_name}")
                except:
                    pass  # Collection might not exist
            
            # Try to get existing collection
            try:
                collection = self.client.get_collection(self.collection_name)
                print(f"Found existing collection: {self.collection_name}")
                
                # Check if collection is compatible with current embedding method
                compatibility_result = self._check_collection_compatibility(collection)
                
                if not compatibility_result["compatible"]:
                    print(f"Collection dimension incompatible: Expected {self.expected_dimension}, "
                          f"found {compatibility_result['existing_dimension']}")
                    print("Creating new collection with compatible dimensions...")
                    
                    # Create backup name with timestamp
                    import time
                    backup_name = f"{self.collection_name}_backup_{int(time.time())}"
                    
                    try:
                        # Try to rename existing collection as backup
                        print(f"Backing up existing collection as: {backup_name}")
                        # Note: ChromaDB doesn't have rename, so we delete the old one
                        self.client.delete_collection(self.collection_name)
                    except Exception as e:
                        print(f"Could not backup collection: {e}")
                    
                    collection = self.client.create_collection(self.collection_name)
                    print(f"Created new collection: {self.collection_name}")
                else:
                    print(f"Collection is compatible with {self.embedding_method} embeddings")
                    # Load existing documents for TF-IDF
                    self._load_existing_documents(collection)
                
                return collection
                
            except Exception as e:
                print(f"Collection not found, creating new: {self.collection_name}")
                return self.client.create_collection(self.collection_name)
                
        except Exception as e:
            print(f"Error initializing collection: {e}")
            raise
    
    def _check_collection_compatibility(self, collection) -> dict:
        """Check if existing collection is compatible with current embedding method"""
        try:
            # Get a small sample of existing documents to check embedding dimension
            existing_docs = collection.get(limit=1)
            
            if not existing_docs or not existing_docs.get("documents") or not existing_docs.get("embeddings"):
                return {"compatible": True, "existing_dimension": None, "reason": "Empty collection"}
            
            # Check actual embedding dimension from stored data
            if existing_docs["embeddings"]:
                existing_dimension = len(existing_docs["embeddings"][0])
                
                # Compare with expected dimension
                if existing_dimension == self.expected_dimension:
                    return {
                        "compatible": True, 
                        "existing_dimension": existing_dimension,
                        "reason": "Dimensions match"
                    }
                else:
                    return {
                        "compatible": False,
                        "existing_dimension": existing_dimension,
                        "expected_dimension": self.expected_dimension,
                        "reason": f"Dimension mismatch: expected {self.expected_dimension}, got {existing_dimension}"
                    }
            
            # Fallback: try to generate test embedding and query
            test_embedding = self.embedding_function(["test document"])
            if not test_embedding:
                return {"compatible": False, "existing_dimension": None, "reason": "Could not generate test embedding"}
                
            test_dimension = len(test_embedding[0])
            
            try:
                # Try a test query to see if dimensions are compatible
                collection.query(
                    query_embeddings=[test_embedding[0]],
                    n_results=1
                )
                return {
                    "compatible": True,
                    "existing_dimension": test_dimension,
                    "reason": "Test query succeeded"
                }
            except Exception as query_error:
                if "dimension" in str(query_error).lower():
                    return {
                        "compatible": False,
                        "existing_dimension": "unknown",
                        "expected_dimension": test_dimension,
                        "reason": f"Dimension error in test query: {query_error}"
                    }
                else:
                    # Other error, might still be compatible
                    return {
                        "compatible": True,
                        "existing_dimension": test_dimension,
                        "reason": f"Non-dimension error in test query: {query_error}"
                    }
                
        except Exception as e:
            print(f"Error checking collection compatibility: {e}")
            return {"compatible": False, "existing_dimension": None, "reason": f"Error: {e}"}
    
    def _load_existing_documents(self, collection):
        """Load existing documents from ChromaDB for TF-IDF refitting"""
        try:
            if self.embedding_method == "tfidf":
                existing_docs = collection.get()
                if existing_docs and existing_docs.get("documents"):
                    self.document_store = existing_docs["documents"]
                    if self.document_store:
                        # Refit vectorizer with existing documents
                        self.vectorizer.fit(self.document_store)
                        self.fitted_vectorizer = True
                        print(f"Loaded {len(self.document_store)} existing documents for TF-IDF")
        except Exception as e:
            print(f"Error loading existing documents: {e}")
            # Reset document store on error
            self.document_store = []
            self.fitted_vectorizer = False
    
    def _get_tfidf_embedding(self, texts: List[str]) -> List[List[float]]:
        """Get TF-IDF embeddings for texts with fixed dimension"""
        try:
            if not self.fitted_vectorizer:
                # Fit vectorizer on all documents (existing + new)
                all_texts = self.document_store + texts
                if not all_texts:
                    # If no texts, create a minimal vocabulary
                    all_texts = ["sample text for initialization"]
                
                self.vectorizer.fit(all_texts)
                self.fitted_vectorizer = True
                print(f"Fitted TF-IDF vectorizer with {len(all_texts)} documents")
            
            # Transform texts to embeddings
            embeddings = self.vectorizer.transform(texts)
            embeddings_array = embeddings.toarray()
            
            # Verify dimension matches expected
            if embeddings_array.shape[1] != self.expected_dimension:
                print(f"Warning: TF-IDF dimension mismatch. Expected {self.expected_dimension}, got {embeddings_array.shape[1]}")
                # Adjust vectorizer max_features if needed
                if embeddings_array.shape[1] < self.expected_dimension:
                    print("Padding embeddings to expected dimension")
                    padded = np.zeros((embeddings_array.shape[0], self.expected_dimension))
                    padded[:, :embeddings_array.shape[1]] = embeddings_array
                    embeddings_array = padded
                else:
                    print("Truncating embeddings to expected dimension")
                    embeddings_array = embeddings_array[:, :self.expected_dimension]
            
            return embeddings_array.tolist()
            
        except Exception as e:
            print(f"Error in TF-IDF embedding: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * self.expected_dimension for _ in texts]
    
    def _get_openai_embedding(self, texts: List[str]) -> List[List[float]]:
        """Get OpenAI embeddings for texts"""
        embeddings = []
        for text in texts:
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = response.data[0].embedding
                
                # Verify dimension
                if len(embedding) != self.expected_dimension:
                    print(f"Warning: OpenAI embedding dimension mismatch. Expected {self.expected_dimension}, got {len(embedding)}")
                    # Truncate or pad as needed
                    if len(embedding) > self.expected_dimension:
                        embedding = embedding[:self.expected_dimension]
                    else:
                        embedding.extend([0.0] * (self.expected_dimension - len(embedding)))
                
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error getting OpenAI embedding: {e}")
                # Fallback to zero vector with correct dimension
                embeddings.append([0.0] * self.expected_dimension)
        
        return embeddings
    
    def _get_custom_embedding(self, texts: List[str]) -> List[List[float]]:
        """Custom embedding function with guaranteed fixed dimension"""
        embeddings = []
        
        # Create a more consistent vocabulary approach
        all_words = set()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.update(words)
        
        # Use a consistent vocabulary (could be pre-defined or learned)
        if hasattr(self, 'custom_vocab'):
            vocab_list = self.custom_vocab
        else:
            # Create vocabulary from current texts, but ensure consistent size
            vocab_list = sorted(list(all_words))[:self.expected_dimension]
            # Pad vocabulary if needed
            while len(vocab_list) < self.expected_dimension:
                vocab_list.append(f"pad_token_{len(vocab_list)}")
            vocab_list = vocab_list[:self.expected_dimension]  # Ensure exact size
            self.custom_vocab = vocab_list  # Cache for consistency
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Create embedding vector with exact expected dimension
            embedding = []
            for word in vocab_list:
                embedding.append(float(word_counts.get(word, 0)))
            
            # Ensure exact dimension
            assert len(embedding) == self.expected_dimension, f"Custom embedding dimension error: got {len(embedding)}, expected {self.expected_dimension}"
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            embeddings.append(embedding)
        
        return embeddings
    
    def reset_collection(self):
        """Reset the collection (delete and recreate)"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
            
            self.collection = self.client.create_collection(self.collection_name)
            print(f"Created new collection: {self.collection_name}")
            
            # Reset document store for TF-IDF
            self.document_store = []
            self.fitted_vectorizer = False
            
            # Reset custom vocabulary if using custom embeddings
            if hasattr(self, 'custom_vocab'):
                delattr(self, 'custom_vocab')
            
            return f"Collection {self.collection_name} has been reset"
            
        except Exception as e:
            error_msg = f"Error resetting collection: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def normalize_filename(self, filename: str) -> str:
        """Normalize filename for consistent storage and retrieval"""
        # Remove path separators and normalize
        normalized = filename.replace('\\', '/').split('/')[-1]
        # Remove extra spaces and normalize case
        normalized = normalized.strip()
        return normalized
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text
                print(f"Extracted {len(page_text)} characters from page {page_num + 1}")
            
            print(f"Total extracted text length: {len(text)}")
            return text
        except Exception as e:
            print(f"Error extracting PDF text: {str(e)}")
            raise
    
    def add_document(self, content: str, filename: str) -> str:
        """Add document to vector store with normalized filename and dimension verification"""
        try:
            if not content.strip():
                raise ValueError("Document content is empty")
            
            # Normalize filename
            normalized_filename = self.normalize_filename(filename)
            logger.info(f"Adding document with normalized filename: {normalized_filename}")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            print(f"Split document into {len(chunks)} chunks")
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Add chunks to document store for TF-IDF
            if self.embedding_method == "tfidf":
                self.document_store.extend(chunks)
                # Refit vectorizer with new documents
                self.vectorizer.fit(self.document_store)
                self.fitted_vectorizer = True
            
            # Generate embeddings
            print(f"Generating embeddings using {self.embedding_method}...")
            embeddings = self.embedding_function(chunks)
            
            # Verify embedding dimensions
            if embeddings:
                embedding_dim = len(embeddings[0])
                print(f"Generated embeddings with dimension: {embedding_dim}")
                
                if embedding_dim != self.expected_dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.expected_dimension}, "
                        f"got {embedding_dim}. This usually means there's an existing collection "
                        f"with different embeddings. Use reset_collection=True to start fresh."
                    )
            else:
                raise ValueError("No embeddings generated")
            
            # Create unique IDs for chunks
            ids = [str(uuid.uuid4()) for _ in chunks]
            
            # Metadata for each chunk with normalized filename
            metadatas = [
                {
                    "filename": normalized_filename,
                    "original_filename": filename,  # Keep original for reference
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "embedding_method": self.embedding_method,
                    "embedding_dimension": embedding_dim
                } 
                for i in range(len(chunks))
            ]
            
            # Add to ChromaDB with dimension verification
            print(f"Adding {len(chunks)} chunks to ChromaDB...")
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            result = f"Successfully added {len(chunks)} chunks from {normalized_filename} using {self.embedding_method} embeddings (dim: {embedding_dim})"
            print(result)
            print(f"Collection now has {self.collection.count()} total documents")
            return result
            
        except Exception as e:
            error_msg = f"Error adding document {filename}: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def add_pdf_document(self, pdf_content: bytes, filename: str) -> str:
        """Add PDF document to vector store"""
        try:
            text_content = self.extract_text_from_pdf(pdf_content)
            if not text_content.strip():
                raise ValueError(f"No text extracted from PDF: {filename}")
            return self.add_document(text_content, filename)
        except Exception as e:
            error_msg = f"Error processing PDF {filename}: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    # ... (rest of the methods remain the same as in the original code)
    # Including: get_available_documents, find_document_by_name, get_document_content,
    # search_similar_enhanced, extract_key_terms, generate_alternative_queries,
    # combine_search_results, rerank_by_keywords, search_similar,
    # get_context_for_query, debug_collection_info
    
    def get_available_documents(self) -> List[Dict[str, str]]:
        """Get list of all available documents with both normalized and original filenames"""
        try:
            search_results = self.collection.get()
            if not search_results or not search_results.get("metadatas"):
                return []
            
            # Extract unique filenames with metadata
            documents = {}
            for metadata in search_results["metadatas"]:
                if metadata and "filename" in metadata:
                    filename = metadata["filename"]
                    original_filename = metadata.get("original_filename", filename)
                    chunk_count = documents.get(filename, {}).get("chunk_count", 0) + 1
                    
                    documents[filename] = {
                        "filename": filename,
                        "original_filename": original_filename,
                        "chunk_count": chunk_count,
                        "embedding_method": metadata.get("embedding_method", "unknown"),
                        "embedding_dimension": metadata.get("embedding_dimension", "unknown")
                    }
            
            # Convert to list and sort
            doc_list = list(documents.values())
            doc_list.sort(key=lambda x: x["filename"].lower())
            
            logger.info(f"Available documents: {[doc['filename'] for doc in doc_list]}")
            return doc_list
            
        except Exception as e:
            logger.error(f"Error getting available documents: {str(e)}")
            return []
    
    def debug_collection_info(self) -> Dict:
        """Debug method to get information about the collection"""
        try:
            print(f"\n=== COLLECTION DEBUG INFO ===")
            print(f"Embedding method: {self.embedding_method}")
            print(f"Expected dimension: {self.expected_dimension}")
            
            doc_count = self.collection.count()
            print(f"Total documents in collection: {doc_count}")
            
            if doc_count == 0:
                return {
                    "total_docs": 0, 
                    "files": [], 
                    "sample_content": [],
                    "embedding_method": self.embedding_method,
                    "expected_dimension": self.expected_dimension
                }
            
            all_docs = self.collection.get()
            
            # Check embedding dimensions in stored data
            actual_dimensions = set()
            if all_docs.get('embeddings'):
                for emb in all_docs['embeddings'][:5]:  # Check first 5
                    if emb:
                        actual_dimensions.add(len(emb))
            
            filenames = set()
            for meta in all_docs.get('metadatas', []):
                if meta and 'filename' in meta:
                    filenames.add(meta['filename'])
            
            print(f"Unique files: {len(filenames)}")
            print(f"Actual embedding dimensions found: {actual_dimensions}")
            
            for filename in sorted(filenames):
                print(f"  - {filename}")
            
            file_samples = {}
            for i, (doc, meta) in enumerate(zip(all_docs.get('documents', []), all_docs.get('metadatas', []))):
                if meta and 'filename' in meta:
                    filename = meta['filename']
                    if filename not in file_samples:
                        file_samples[filename] = {
                            'chunk_count': 0,
                            'sample_content': doc[:300] + "..." if len(doc) > 300 else doc,
                            'embedding_method': meta.get('embedding_method', 'unknown'),
                            'embedding_dimension': meta.get('embedding_dimension', 'unknown')
                        }
                    file_samples[filename]['chunk_count'] += 1
            
            print(f"\nFile details:")
            for filename, info in file_samples.items():
                print(f"  {filename}: {info['chunk_count']} chunks "
                      f"({info['embedding_method']} embeddings, dim: {info['embedding_dimension']})")
                print(f"    Sample: {info['sample_content'][:100]}...")
                print()
            
            return {
                "total_docs": doc_count,
                "files": list(filenames),
                "file_details": file_samples,
                "sample_content": [doc[:200] for doc in all_docs.get('documents', [])[:3]],
                "embedding_method": self.embedding_method,
                "expected_dimension": self.expected_dimension,
                "actual_dimensions": list(actual_dimensions)
            }
            
        except Exception as e:
            print(f"Error in debug_collection_info: {str(e)}")
            return {"error": str(e)}
    
    # Add the remaining methods from the original code...
    def find_document_by_name(self, filename: str) -> Optional[str]:
        """Find document by various matching strategies"""
        try:
            available_docs = self.get_available_documents()
            
            if not available_docs:
                logger.warning("No documents available in collection")
                return None
            
            # Normalize the search filename
            search_filename = self.normalize_filename(filename)
            logger.info(f"Searching for document: '{search_filename}'")
            
            # Strategy 1: Exact match (normalized)
            for doc in available_docs:
                if doc["filename"] == search_filename:
                    logger.info(f"Found exact match: {doc['filename']}")
                    return doc["filename"]
            
            # Strategy 2: Case-insensitive match
            search_lower = search_filename.lower()
            for doc in available_docs:
                if doc["filename"].lower() == search_lower:
                    logger.info(f"Found case-insensitive match: {doc['filename']}")
                    return doc["filename"]
            
            # Strategy 3: Partial match (contains)
            for doc in available_docs:
                if search_lower in doc["filename"].lower() or doc["filename"].lower() in search_lower:
                    logger.info(f"Found partial match: {doc['filename']}")
                    return doc["filename"]
            
            # Strategy 4: Match without extension
            search_no_ext = search_filename.rsplit('.', 1)[0] if '.' in search_filename else search_filename
            for doc in available_docs:
                doc_no_ext = doc["filename"].rsplit('.', 1)[0] if '.' in doc["filename"] else doc["filename"]
                if search_no_ext.lower() == doc_no_ext.lower():
                    logger.info(f"Found extension-agnostic match: {doc['filename']}")
                    return doc["filename"]
            
            # Strategy 5: Match against original filename
            for doc in available_docs:
                if doc.get("original_filename", "").lower() == search_lower:
                    logger.info(f"Found original filename match: {doc['filename']}")
                    return doc["filename"]
            
            logger.warning(f"No match found for '{filename}' among available documents: {[doc['filename'] for doc in available_docs]}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding document by name: {str(e)}")
            return None
    
    def get_context_for_query(self, query: str, max_context_length: int = 3000) -> tuple:
        """Get relevant context with improved search"""
        try:
            search_results = self.search_similar_enhanced(query, n_results=10)
            
            if not search_results["documents"]:
                print("No search results found")
                return "", []
            
            context_chunks = []
            sources = set()
            total_length = 0
            
            sorted_results = sorted(
                zip(search_results["documents"], search_results["metadatas"], search_results["distances"]),
                key=lambda x: x[2]
            )
            
            print(f"Processing {len(sorted_results)} search results")
            
            for i, (doc, metadata, distance) in enumerate(sorted_results):
                if total_length + len(doc) <= max_context_length:
                    context_chunks.append(doc)
                    sources.add(metadata["filename"])
                    total_length += len(doc)
                    print(f"  Added chunk {i+1} from {metadata['filename']} (distance: {distance:.4f})")
                else:
                    break
            
            context = "\n\n".join(context_chunks)
            print(f"Built context with {len(context_chunks)} chunks, total length: {len(context)}")
            
            return context, list(sources)
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            return "", []
    
    def search_similar_enhanced(self, query: str, n_results: int = 10) -> Dict:
        """Enhanced search with multiple strategies"""
        try:
            doc_count = self.collection.count()
            if doc_count == 0:
                print("No documents in collection to search")
                return {"documents": [], "metadatas": [], "distances": []}
            
            print(f"Enhanced search for: '{query}' using {self.embedding_method} embeddings")
            
            # Generate query embedding with dimension verification
            query_embedding = self.embedding_function([query])[0]
            if len(query_embedding) != self.expected_dimension:
                raise ValueError(f"Query embedding dimension mismatch: expected {self.expected_dimension}, got {len(query_embedding)}")
            
            actual_n_results = min(n_results, doc_count)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_n_results
            )
            
            # Strategy 2: Keyword-based filtering for better results
            key_terms = self.extract_key_terms(query)
            print(f"Key terms extracted: {key_terms}")
            
            # Strategy 3: Try alternative phrasings
            alternative_queries = self.generate_alternative_queries(query)
            print(f"Alternative queries: {alternative_queries}")
            
            # Combine results from different strategies
            combined_results = self.combine_search_results(results, query, key_terms, alternative_queries)
            
            print(f"Enhanced search returned {len(combined_results['documents'])} results")
            
            return combined_results
            
        except Exception as e:
            print(f"Error in enhanced search: {str(e)}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for better matching"""
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                     'to', 'was', 'will', 'with', 'few', 'things', 'does', 'not', 'do'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def generate_alternative_queries(self, original_query: str) -> List[str]:
        """Generate alternative query phrasings"""
        alternatives = []
        
        if "does not do" in original_query.lower():
            alternatives.extend([
                "what science cannot do",
                "limitations of science",
                "science limitations",
                "what science doesn't do",
                "boundaries of science",
                "science cannot",
                "not scientific"
            ])
        
        if "science" in original_query.lower():
            alternatives.extend([
                "scientific method",
                "scientific approach",
                "scientific process"
            ])
        
        return alternatives
    
    def combine_search_results(self, primary_results: Dict, original_query: str, 
                             key_terms: List[str], alternatives: List[str]) -> Dict:
        """Combine and rerank search results from multiple strategies"""
        all_results = {
            "documents": primary_results["documents"][0] if primary_results["documents"] else [],
            "metadatas": primary_results["metadatas"][0] if primary_results["metadatas"] else [],
            "distances": primary_results["distances"][0] if primary_results["distances"] else []
        }
        
        # Try alternative queries if primary results are poor
        if not all_results["documents"] or (all_results["distances"] and min(all_results["distances"]) > 0.7):
            print("Primary search results poor, trying alternatives...")
            
            for alt_query in alternatives[:3]:
                try:
                    alt_embedding = self.embedding_function([alt_query])[0]
                    alt_results = self.collection.query(
                        query_embeddings=[alt_embedding],
                        n_results=5
                    )
                    
                    if alt_results["documents"] and alt_results["documents"][0]:
                        print(f"Alternative query '{alt_query}' found {len(alt_results['documents'][0])} results")
                        
                        for doc, meta, dist in zip(
                            alt_results["documents"][0],
                            alt_results["metadatas"][0], 
                            alt_results["distances"][0]
                        ):
                            if doc not in all_results["documents"]:
                                all_results["documents"].append(doc)
                                all_results["metadatas"].append(meta)
                                all_results["distances"].append(dist + 0.1)
                                
                except Exception as e:
                    print(f"Error with alternative query '{alt_query}': {e}")
                    continue
        
        # Rerank by keyword presence
        if key_terms:
            return self.rerank_by_keywords(all_results, key_terms)
        
        return all_results
    
    def rerank_by_keywords(self, results: Dict, key_terms: List[str]) -> Dict:
        """Rerank results based on keyword presence"""
        if not results["documents"]:
            return results
        
        scored_results = []
        for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
            keyword_score = sum(1 for term in key_terms if term.lower() in doc.lower())
            combined_score = dist - (keyword_score * 0.1)
            scored_results.append((doc, meta, dist, combined_score))
        
        scored_results.sort(key=lambda x: x[3])
        
        return {
            "documents": [x[0] for x in scored_results],
            "metadatas": [x[1] for x in scored_results],
            "distances": [x[2] for x in scored_results]
        }
    
    def search_similar(self, query: str, n_results: int = 5) -> Dict:
        """Main search method - now uses enhanced search"""
        return self.search_similar_enhanced(query, n_results)
    
    def get_context_for_query(self, query: str, max_context_length: int = 3000) -> tuple:
        """Get relevant context with improved search"""
        try:
            search_results = self.search_similar_enhanced(query, n_results=10)
            
            if not search_results["documents"]:
                print("No search results found")
                return "", []
            
            context_chunks = []
            sources = set()
            total_length = 0
            
            sorted_results = sorted(
                zip(search_results["documents"], search_results["metadatas"], search_results["distances"]),
                key=lambda x: x[2]
            )
            
            print(f"Processing {len(sorted_results)} search results")
            
            for i, (doc, metadata, distance) in enumerate(sorted_results):
                if total_length + len(doc) <= max_context_length:
                    context_chunks.append(doc)
                    sources.add(metadata["filename"])
                    total_length += len(doc)
                    print(f"  Added chunk {i+1} from {metadata['filename']} (distance: {distance:.4f})")
                else:
                    break
            
            context = "\n\n".join(context_chunks)
            print(f"Built context with {len(context_chunks)} chunks, total length: {len(context)}")
            
            return context, list(sources)
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            return "", []
    
    def debug_collection_info(self) -> Dict:
        """Debug method to get information about the collection"""
        try:
            print(f"\n=== COLLECTION DEBUG INFO ===")
            print(f"Embedding method: {self.embedding_method}")
            
            doc_count = self.collection.count()
            print(f"Total documents in collection: {doc_count}")
            
            if doc_count == 0:
                return {"total_docs": 0, "files": [], "sample_content": []}
            
            all_docs = self.collection.get()
            
            filenames = set()
            for meta in all_docs.get('metadatas', []):
                if meta and 'filename' in meta:
                    filenames.add(meta['filename'])
            
            print(f"Unique files: {len(filenames)}")
            for filename in sorted(filenames):
                print(f"  - {filename}")
            
            file_samples = {}
            for i, (doc, meta) in enumerate(zip(all_docs.get('documents', []), all_docs.get('metadatas', []))):
                if meta and 'filename' in meta:
                    filename = meta['filename']
                    if filename not in file_samples:
                        file_samples[filename] = {
                            'chunk_count': 0,
                            'sample_content': doc[:300] + "..." if len(doc) > 300 else doc,
                            'embedding_method': meta.get('embedding_method', 'unknown'),
                            'embedding_dimension': meta.get('embedding_dimension', 'unknown')
                        }
                    file_samples[filename]['chunk_count'] += 1
            
            print(f"\nFile details:")
            for filename, info in file_samples.items():
                print(f"  {filename}: {info['chunk_count']} chunks ({info['embedding_method']} embeddings, dim: {info['embedding_dimension']})")
                print(f"    Sample: {info['sample_content'][:100]}...")
                print()
            
            return {
                "total_docs": doc_count,
                "files": list(filenames),
                "file_details": file_samples,
                "sample_content": [doc[:200] for doc in all_docs.get('documents', [])[:3]],
                "embedding_method": self.embedding_method
            }
            
        except Exception as e:
            print(f"Error in debug_collection_info: {str(e)}")
            return {"error": str(e)}