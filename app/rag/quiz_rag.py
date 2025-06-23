from typing import List, Dict, Any, Optional
import os
# Fixed imports to address deprecation warnings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import tempfile
import logging

logger = logging.getLogger(__name__)

class QuizRAGService:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize Quiz RAG Service with embeddings
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{embedding_model}"
        )
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
        Create vector store from processed documents
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
                    valid_documents.append(doc)
                else:
                    logger.warning(f"Skipping invalid document type: {type(doc)}")
            
            if not valid_documents:
                logger.error("No valid documents found for vector store creation")
                return False
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=valid_documents,
                embedding=self.embeddings
            )
            
            logger.info(f"Created vector store with {len(valid_documents)} documents")
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
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.vector_store = None
        self.documents = []
        logger.info("Quiz RAG resources cleaned up")