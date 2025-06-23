# Improved RAG System with Better Search Strategies
import chromadb
import PyPDF2
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import uuid
import io
import re

class RAGSystem:
    def __init__(self, collection_name: str = "ai_tutor_docs"):
        # Initialize ChromaDB with new configuration
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Print initial state
        doc_count = self.collection.count()
        print(f"Collection initialized with {doc_count} documents")
    
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
        """Add document to vector store"""
        try:
            if not content.strip():
                raise ValueError("Document content is empty")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            print(f"Split document into {len(chunks)} chunks")
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Create unique IDs for chunks
            ids = [str(uuid.uuid4()) for _ in chunks]
            
            # Metadata for each chunk
            metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            result = f"Added {len(chunks)} chunks from {filename}"
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
    
    def search_similar_enhanced(self, query: str, n_results: int = 10) -> Dict:
        """Enhanced search with multiple strategies"""
        try:
            doc_count = self.collection.count()
            if doc_count == 0:
                print("No documents in collection to search")
                return {"documents": [], "metadatas": [], "distances": []}
            
            print(f"Enhanced search for: '{query}'")
            
            # Strategy 1: Direct semantic search
            query_embedding = self.embedding_model.encode([query]).tolist()
            actual_n_results = min(n_results, doc_count)
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=actual_n_results
            )
            
            # Strategy 2: Keyword-based filtering for better results
            # Extract key terms from query
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
        # Remove common stop words and extract meaningful terms
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                     'to', 'was', 'will', 'with', 'few', 'things', 'does', 'not', 'do'}
        
        # Clean and split query
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def generate_alternative_queries(self, original_query: str) -> List[str]:
        """Generate alternative query phrasings"""
        alternatives = []
        
        # For "A few things that science does not do"
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
        
        # Add more patterns as needed
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
            
            for alt_query in alternatives[:3]:  # Try top 3 alternatives
                try:
                    alt_embedding = self.embedding_model.encode([alt_query]).tolist()
                    alt_results = self.collection.query(
                        query_embeddings=alt_embedding,
                        n_results=5
                    )
                    
                    if alt_results["documents"] and alt_results["documents"][0]:
                        print(f"Alternative query '{alt_query}' found {len(alt_results['documents'][0])} results")
                        
                        # Add unique results
                        for doc, meta, dist in zip(
                            alt_results["documents"][0],
                            alt_results["metadatas"][0], 
                            alt_results["distances"][0]
                        ):
                            if doc not in all_results["documents"]:
                                all_results["documents"].append(doc)
                                all_results["metadatas"].append(meta)
                                all_results["distances"].append(dist + 0.1)  # Slight penalty for alternative query
                                
                except Exception as e:
                    print(f"Error with alternative query '{alt_query}': {e}")
                    continue
        
        # Rerank by keyword presence
        if key_terms:
            reranked_results = self.rerank_by_keywords(all_results, key_terms)
            return reranked_results
        
        return all_results
    
    def rerank_by_keywords(self, results: Dict, key_terms: List[str]) -> Dict:
        """Rerank results based on keyword presence"""
        if not results["documents"]:
            return results
        
        # Calculate keyword scores
        scored_results = []
        for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
            keyword_score = sum(1 for term in key_terms if term.lower() in doc.lower())
            # Combine semantic similarity with keyword relevance
            combined_score = dist - (keyword_score * 0.1)  # Lower is better
            scored_results.append((doc, meta, dist, combined_score))
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x[3])
        
        # Return reranked results
        return {
            "documents": [x[0] for x in scored_results],
            "metadatas": [x[1] for x in scored_results],
            "distances": [x[2] for x in scored_results]
        }
    
    def search_similar(self, query: str, n_results: int = 5) -> Dict:
        """Main search method - now uses enhanced search"""
        return self.search_similar_enhanced(query, n_results)
    
    def get_context_for_query(self, query: str, max_context_length: int = 3000) -> tuple:
        """Get relevant context with improved search and larger context window"""
        try:
            search_results = self.search_similar_enhanced(query, n_results=10)
            
            if not search_results["documents"]:
                print("No search results found")
                return "", []
            
            context_chunks = []
            sources = set()
            total_length = 0
            
            # Prioritize results with better scores (lower distances)
            sorted_results = sorted(
                zip(search_results["documents"], search_results["metadatas"], search_results["distances"]),
                key=lambda x: x[2]  # Sort by distance
            )
            
            print(f"Processing {len(sorted_results)} search results")
            
            for i, (doc, metadata, distance) in enumerate(sorted_results):
                if total_length + len(doc) <= max_context_length:
                    context_chunks.append(doc)
                    sources.add(metadata["filename"])
                    total_length += len(doc)
                    print(f"  Added chunk {i+1} from {metadata['filename']} (distance: {distance:.4f}, length: {len(doc)})")
                else:
                    print(f"  Skipped chunk {i+1} - would exceed max context length")
                    break
            
            context = "\n\n".join(context_chunks)
            print(f"Built context with {len(context_chunks)} chunks, total length: {len(context)}")
            
            return context, list(sources)
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            return "", []
    
    def debug_search_for_query(self, query: str) -> Dict:
        """Debug method to understand why search results might not be optimal"""
        try:
            print(f"\n=== DEBUG SEARCH FOR: '{query}' ===")
            
            # Get all documents and search through them manually
            all_docs = self.collection.get()
            
            # Find documents containing key terms
            key_terms = self.extract_key_terms(query)
            print(f"Key terms: {key_terms}")
            
            matches = []
            for i, (doc, meta) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                score = 0
                matched_terms = []
                
                for term in key_terms:
                    if term.lower() in doc.lower():
                        score += 1
                        matched_terms.append(term)
                
                # Also check for the full query or parts of it
                if query.lower() in doc.lower():
                    score += 5
                    matched_terms.append("FULL_QUERY")
                
                if score > 0:
                    matches.append({
                        "chunk_index": i,
                        "filename": meta.get('filename', 'Unknown'),
                        "score": score,
                        "matched_terms": matched_terms,
                        "content_preview": doc[:200] + "..."
                    })
            
            # Sort by score
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"Found {len(matches)} chunks with keyword matches:")
            for match in matches[:5]:
                print(f"  Score: {match['score']}, Terms: {match['matched_terms']}")
                print(f"  Content: {match['content_preview']}")
                print()
            
            return {
                "query": query,
                "key_terms": key_terms,
                "matches": matches[:10]
            }
            
        except Exception as e:
            print(f"Error in debug search: {str(e)}")
            return {"error": str(e)}
        
        
    def debug_collection_info(self) -> Dict:
     """Debug method to get information about the collection"""
     try:
        print(f"\n=== COLLECTION DEBUG INFO ===")
        
        # Get basic collection stats
        doc_count = self.collection.count()
        print(f"Total documents in collection: {doc_count}")
        
        if doc_count == 0:
            return {"total_docs": 0, "files": [], "sample_content": []}
        
        # Get all documents
        all_docs = self.collection.get()
        
        # Get unique filenames
        filenames = set()
        for meta in all_docs.get('metadatas', []):
            if meta and 'filename' in meta:
                filenames.add(meta['filename'])
        
        print(f"Unique files: {len(filenames)}")
        for filename in sorted(filenames):
            print(f"  - {filename}")
        
        # Get sample content from each file
        file_samples = {}
        for i, (doc, meta) in enumerate(zip(all_docs.get('documents', []), all_docs.get('metadatas', []))):
            if meta and 'filename' in meta:
                filename = meta['filename']
                if filename not in file_samples:
                    file_samples[filename] = {
                        'chunk_count': 0,
                        'sample_content': doc[:300] + "..." if len(doc) > 300 else doc
                    }
                file_samples[filename]['chunk_count'] += 1
        
        print(f"\nFile details:")
        for filename, info in file_samples.items():
            print(f"  {filename}: {info['chunk_count']} chunks")
            print(f"    Sample: {info['sample_content'][:100]}...")
            print()
        
        return {
            "total_docs": doc_count,
            "files": list(filenames),
            "file_details": file_samples,
            "sample_content": [doc[:200] for doc in all_docs.get('documents', [])[:3]]
        }
        
     except Exception as e:
        print(f"Error in debug_collection_info: {str(e)}")
        return {"error": str(e)}    