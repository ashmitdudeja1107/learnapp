from typing import List, Dict, Any, Optional
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import tempfile
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class TFIDFEmbeddings(Embeddings):
    """
    TF-IDF based embeddings implementation with adaptive parameters
    """
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        super().__init__()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.is_fitted = False
    
    def _create_vectorizer(self, num_documents: int):
        """Create vectorizer with adaptive parameters based on document count"""
        # Adaptive parameters based on document count
        if num_documents < 5:
            # Very few documents - use minimal constraints
            min_df = 1
            max_df = 1.0
            max_features = min(1000, self.max_features)
        elif num_documents < 20:
            # Small collection - relax constraints
            min_df = 1
            max_df = 0.9
            max_features = min(2000, self.max_features)
        else:
            # Larger collection - use original constraints
            min_df = 2
            max_df = 0.95
            max_features = self.max_features
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=self.ngram_range,
            max_df=max_df,
            min_df=min_df
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if not texts:
            return []
        
        # Create vectorizer with adaptive parameters
        self._create_vectorizer(len(texts))
        
        try:
            if not self.is_fitted:
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                self.is_fitted = True
            else:
                tfidf_matrix = self.vectorizer.transform(texts)
            
            # Convert sparse matrix to dense and then to list of lists
            return tfidf_matrix.toarray().tolist()
        
        except ValueError as e:
            # Fallback: create a simple vectorizer with minimal constraints
            logger.warning(f"TF-IDF vectorizer failed with error: {e}. Using fallback vectorizer.")
            
            self.vectorizer = TfidfVectorizer(
                max_features=min(1000, self.max_features),
                stop_words='english',
                ngram_range=(1, 1),  # Only unigrams
                max_df=1.0,
                min_df=1
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
            return tfidf_matrix.toarray().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not self.is_fitted:
            raise ValueError("Embeddings not fitted yet. Call embed_documents first.")
        
        query_vector = self.vectorizer.transform([text])
        return query_vector.toarray().tolist()[0]

class QuizRAGService:
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Initialize Quiz RAG Service with TF-IDF embeddings
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams for TF-IDF
        """
        self.embeddings = TFIDFEmbeddings(max_features=max_features, ngram_range=ngram_range)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = None
        self.documents = []
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document based on file extension
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []
    
    def process_document_for_quiz(self, file_path: str) -> List[Document]:
        """
        Process documents specifically for quiz generation - returns processed documents
        """
        try:
            # Load documents
            raw_documents = self.load_document(file_path)
        
            if not raw_documents:
                logger.warning("No documents loaded from file")
                return []
        
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(raw_documents)
        
            # Filter out very short chunks that won't be useful for quiz generation
            filtered_chunks = [
                chunk for chunk in chunks 
                if len(chunk.page_content.strip()) > 100
            ]
        
            if not filtered_chunks:
                logger.warning("No suitable content chunks found for quiz generation")
                return []
        
            self.documents = filtered_chunks
            logger.info(f"Processed {len(filtered_chunks)} chunks for quiz generation")
        
            return filtered_chunks
        
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return []
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        """
        Create vector store from processed documents with better error handling
        """
        try:
            if not documents:
                logger.warning("No documents provided for vector store creation")
                return False
            
            # Ensure documents is a list of Document objects
            if not isinstance(documents, list):
                logger.error(f"Expected list of documents, got {type(documents)}")
                return False
            
            # Validate that all items are Document objects
            valid_documents = []
            for doc in documents:
                if isinstance(doc, Document):
                    if len(doc.page_content.strip()) > 10:  # Ensure meaningful content
                        valid_documents.append(doc)
                else:
                    logger.warning(f"Skipping invalid document type: {type(doc)}")
            
            if not valid_documents:
                logger.error("No valid documents found for vector store creation")
                return False
            
            # Check if we have enough content for meaningful embeddings
            if len(valid_documents) < 2:
                logger.warning(f"Only {len(valid_documents)} document(s) available. This may affect embedding quality.")
            
            # Create FAISS vector store
            try:
                self.vector_store = FAISS.from_documents(
                    documents=valid_documents,
                    embedding=self.embeddings
                )
                
                logger.info(f"Created vector store with {len(valid_documents)} documents")
                return True
                
            except Exception as embedding_error:
                logger.error(f"Error in embedding creation: {str(embedding_error)}")
                
                # Try with a simpler embedding approach if needed
                logger.info("Attempting to recreate embeddings with simpler parameters...")
                
                # Reinitialize with simpler parameters
                self.embeddings = TFIDFEmbeddings(max_features=500, ngram_range=(1, 1))
                
                self.vector_store = FAISS.from_documents(
                    documents=valid_documents,
                    embedding=self.embeddings
                )
                
                logger.info(f"Successfully created vector store with fallback embeddings")
                return True
                
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant document chunks for a query
        """
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return []
            
            # Perform similarity search
            relevant_docs = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {str(e)}")
            return []
    
    def get_diverse_chunks(self, num_chunks: int = 5) -> List[Document]:
        """
        Get diverse chunks from the document for varied quiz questions
        """
        try:
            if not self.documents:
                return []
            
            # If we have fewer documents than requested, return all
            if len(self.documents) <= num_chunks:
                return self.documents
            
            # Use simple strategy to get diverse chunks
            # Divide documents into sections and pick from each
            total_docs = len(self.documents)
            step = max(1, total_docs // num_chunks)
            
            diverse_chunks = []
            for i in range(0, total_docs, step):
                if len(diverse_chunks) < num_chunks:
                    diverse_chunks.append(self.documents[i])
            
            # If we still need more chunks, add random ones
            remaining_docs = [doc for doc in self.documents if doc not in diverse_chunks]
            while len(diverse_chunks) < num_chunks and remaining_docs:
                diverse_chunks.append(remaining_docs.pop(0))
            
            return diverse_chunks[:num_chunks]
            
        except Exception as e:
            logger.error(f"Error getting diverse chunks: {str(e)}")
            return []
    
    def get_context_for_question_generation(self, topic_hint: str = None, chunk_size: int = 800) -> List[str]:
        """
        Get contextual information for question generation
        """
        try:
            contexts = []
            
            if topic_hint and self.vector_store:
                # Get topic-specific chunks
                relevant_chunks = self.retrieve_relevant_chunks(topic_hint, k=3)
                for chunk in relevant_chunks:
                    context = chunk.page_content[:chunk_size]
                    contexts.append(context)
            
            # Also get some diverse chunks
            diverse_chunks = self.get_diverse_chunks(3)
            for chunk in diverse_chunks:
                context = chunk.page_content[:chunk_size]
                if context not in contexts:  # Avoid duplicates
                    contexts.append(context)
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error getting context for question generation: {str(e)}")
            return []
    
    def setup_quiz_rag(self, file_path: str) -> bool:
        """
        Complete setup for quiz RAG system
        """
        try:
            # Process documents - returns List[Document]
            documents = self.process_document_for_quiz(file_path)
            
            if not documents:
                logger.error("No documents processed for quiz RAG")
                return False
            
            # Create vector store - pass the documents list
            success = self.create_vector_store(documents)
            
            if success:
                logger.info("Quiz RAG system successfully initialized")
                return True
            else:
                logger.error("Failed to create vector store")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up quiz RAG: {str(e)}")
            return False
    
    def get_document_summary(self) -> Dict[str, Any]:
        """
        Get summary information about loaded documents
        """
        try:
            if not self.documents:
                return {"total_chunks": 0, "total_content_length": 0}
            
            total_length = sum(len(doc.page_content) for doc in self.documents)
            
            return {
                "total_chunks": len(self.documents),
                "total_content_length": total_length,
                "average_chunk_length": total_length // len(self.documents) if self.documents else 0,
                "has_vector_store": self.vector_store is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting document summary: {str(e)}")
            return {"error": str(e)}
    
    def get_content_chunks_for_quiz(self, num_chunks: int = 5) -> List[str]:
        """
        Get content chunks as strings for quiz generation
        """
        try:
            diverse_chunks = self.get_diverse_chunks(num_chunks)
            return [chunk.page_content for chunk in diverse_chunks]
        except Exception as e:
            logger.error(f"Error getting content chunks for quiz: {str(e)}")
            return []
    
    def search_documents_by_keywords(self, keywords: List[str], max_results: int = 5) -> List[Document]:
        """
        Search documents by keywords without using vector store
        """
        try:
            if not self.documents or not keywords:
                return []
            
            matching_docs = []
            keywords_lower = [kw.lower() for kw in keywords]
            
            for doc in self.documents:
                content_lower = doc.page_content.lower()
                score = sum(1 for kw in keywords_lower if kw in content_lower)
                
                if score > 0:
                    matching_docs.append((doc, score))
            
            # Sort by score (descending) and return top results
            matching_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in matching_docs[:max_results]]
            
        except Exception as e:
            logger.error(f"Error searching documents by keywords: {str(e)}")
            return []
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the loaded documents
        """
        try:
            if not self.documents:
                return {"message": "No documents loaded"}
            
            word_counts = []
            char_counts = []
            
            for doc in self.documents:
                content = doc.page_content
                word_count = len(content.split())
                char_count = len(content)
                
                word_counts.append(word_count)
                char_counts.append(char_count)
            
            return {
                "total_documents": len(self.documents),
                "total_words": sum(word_counts),
                "total_characters": sum(char_counts),
                "average_words_per_chunk": sum(word_counts) / len(word_counts) if word_counts else 0,
                "average_chars_per_chunk": sum(char_counts) / len(char_counts) if char_counts else 0,
                "min_words": min(word_counts) if word_counts else 0,
                "max_words": max(word_counts) if word_counts else 0,
                "vector_store_ready": self.vector_store is not None,
                "embeddings_fitted": self.embeddings.is_fitted if hasattr(self.embeddings, 'is_fitted') else False
            }
            
        except Exception as e:
            logger.error(f"Error getting document statistics: {str(e)}")
            return {"error": str(e)}
    
    def reinitialize_embeddings(self, max_features: int = 1000, ngram_range: tuple = (1, 1)) -> bool:
        """
        Reinitialize embeddings with different parameters (useful for troubleshooting)
        """
        try:
            logger.info(f"Reinitializing embeddings with max_features={max_features}, ngram_range={ngram_range}")
            
            # Create new embeddings instance
            self.embeddings = TFIDFEmbeddings(max_features=max_features, ngram_range=ngram_range)
            
            # Recreate vector store if documents are available
            if self.documents:
                success = self.create_vector_store(self.documents)
                if success:
                    logger.info("Successfully reinitialized embeddings and vector store")
                    return True
                else:
                    logger.error("Failed to recreate vector store with new embeddings")
                    return False
            else:
                logger.info("Embeddings reinitialized, but no documents available for vector store")
                return True
                
        except Exception as e:
            logger.error(f"Error reinitializing embeddings: {str(e)}")
            return False
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.vector_store = None
        self.documents = []
        self.embeddings = TFIDFEmbeddings()
        logger.info("Quiz RAG resources cleaned up")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    quiz_rag = QuizRAGService()
    
    # Example usage
    file_path = "path/to/your/document.pdf"
    
    # Setup the RAG system
    success = quiz_rag.setup_quiz_rag(file_path)
    
    if success:
        print("Quiz RAG system initialized successfully!")
        
        # Get document summary
        summary = quiz_rag.get_document_summary()
        print(f"Document summary: {summary}")
        
        # Get statistics
        stats = quiz_rag.get_document_statistics()
        print(f"Document statistics: {stats}")
        
        # Get content chunks for quiz generation
        chunks = quiz_rag.get_content_chunks_for_quiz(3)
        print(f"Retrieved {len(chunks)} content chunks")
        
        # Search for specific content
        if quiz_rag.vector_store:
            relevant_docs = quiz_rag.retrieve_relevant_chunks("your search query", k=2)
            print(f"Found {len(relevant_docs)} relevant documents")
    
    else:
        print("Failed to initialize Quiz RAG system")
        
        # Try with simpler parameters
        print("Attempting with simpler embedding parameters...")
        success = quiz_rag.reinitialize_embeddings(max_features=500, ngram_range=(1, 1))
        
        if success:
            print("Successfully reinitialized with simpler parameters")
        else:
            print("Failed to initialize even with simpler parameters")
    
    # Cleanup
    quiz_rag.cleanup()