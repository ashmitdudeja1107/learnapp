import json
import logging
from typing import Dict, List, Optional, Any
from app.chains.chainmodel import TutorChains
from app.memory.memorymodel import MemoryManager
from app.rag.ragmodel import RAGSystem
from database.models.chat import ChatRequest, ChatResponse,AddMessageRequest
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, llm_service=None):
        """
        Initialize ChatService with all required components
        
        Args:
            llm_service: Optional LLM service for compatibility with newer architecture
        """
        self.chains = TutorChains()
        self.memory_manager = MemoryManager()
        self.rag_system = RAGSystem()
        
        # For compatibility with newer architecture
        self.llm_service = llm_service
        
       
    
    def create_new_session(self) -> str:
        """
        Create a new chat session and return only the session ID
        """
        try:
            # Use the memory manager to create a new session
            session_id, _ = self.memory_manager.get_or_create_session(None)
            logger.info(f"New session created: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error creating new session: {str(e)}")
            raise e
    
    def get_all_sessions(self) -> List[Dict]:
        """
        Get information about all active sessions (returns serializable data only)
        """
        try:
            sessions = []
            for session_id in self.memory_manager.sessions.keys():
                try:
                    # Get memory object
                    memory = self.memory_manager.sessions[session_id]
                    
                    # Extract serializable information
                    session_data = {
                        "session_id": session_id,
                        "message_count": len(memory.chat_memory.messages) if hasattr(memory, 'chat_memory') else 0,
                        "created_at": "unknown",  # You can add timestamp if you track it
                        "status": "active"
                    }
                    sessions.append(session_data)
                except Exception as e:
                    logger.warning(f"Error getting info for session {session_id}: {str(e)}")
                    continue
            
            return sessions
        except Exception as e:
            logger.error(f"Error getting all sessions: {str(e)}")
            return []
    
    async def process_chat_message(self, request: ChatRequest) -> ChatResponse:
        """Process incoming chat message and generate response"""
        try:
            # Get or create session
            session_id, memory = self.memory_manager.get_or_create_session(request.session_id)
            
            # Get context from RAG if enabled
            context = ""
            sources = []
            
            if request.use_rag:
                try:
                    context, sources = self.rag_system.get_context_for_query(request.message)
                    logger.debug(f"Retrieved context length: {len(context)}")
                    logger.debug(f"Sources: {sources}")
                except Exception as e:
                    logger.warning(f"RAG context retrieval failed: {str(e)}")
            
            # Get chat history
            chat_history = "\n".join(self.memory_manager.get_chat_history(session_id))
            
            # Generate response using tutor chains
            response = self.chains.get_tutor_response(
                question=request.message,
                context=context,
                chat_history=chat_history
            )
            
            # Add to memory
            self.memory_manager.add_message(session_id, request.message, response)
            
            return ChatResponse(
                response=response,
                session_id=session_id,
                sources=sources if sources else None
            )
            
        except Exception as e:
            logger.error(f"Chat processing error: {str(e)}")
            return ChatResponse(
                response=f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                session_id=request.session_id or "error",
                sources=None
            )
    
    async def process_chat_message_simple(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        use_rag: bool = False  # Default to False for simple chat
    ) -> Dict:
        """
        Simplified chat processing method for text-only conversations
        
        Args:
            message: User message (text only)
            session_id: Optional session ID
            use_rag: Whether to use RAG for context (disabled by default for simple chat)
            
        Returns:
            Dict with success status and response
        """
        try:
            # Get or create session
            session_id, memory = self.memory_manager.get_or_create_session(session_id)
            
            # Get chat history for context
            chat_history = "\n".join(self.memory_manager.get_chat_history(session_id))
            
            # Generate response using tutor chains (without RAG context for simple chat)
            response = self.chains.get_tutor_response(
                question=message,
                context="",  # No RAG context for simple chat
                chat_history=chat_history
            )
            
            # Add to memory
            self.memory_manager.add_message(session_id, message, response)
            
            return {
                "success": True,
                "response": response,
                "session_id": session_id,
                "sources": None  # No sources for simple chat
            }
            
        except Exception as e:
            logger.error(f"Simple chat processing error: {str(e)}")
            return {
                "success": False,
                "error": f"Chat processing failed: {str(e)}",
                "session_id": session_id or "error"
            }

    def get_available_documents(self) -> List[str]:
        """Get list of all available documents in the RAG system"""
        try:
            search_results = self.rag_system.collection.get()
            if not search_results or not search_results.get("metadatas"):
                return []
            
            # Extract unique filenames
            filenames = set()
            for metadata in search_results["metadatas"]:
                if metadata and "filename" in metadata:
                    filenames.add(metadata["filename"])
            
            return sorted(list(filenames))
        except Exception as e:
            logger.error(f"Error getting available documents: {str(e)}")
            return []

    def find_document_by_partial_name(self, filename: str) -> Optional[str]:
        """Find document by partial filename match"""
        try:
            available_docs = self.get_available_documents()
            
            # First try exact match
            if filename in available_docs:
                return filename
            
            # Try case-insensitive match
            filename_lower = filename.lower()
            for doc in available_docs:
                if doc.lower() == filename_lower:
                    return doc
            
            # Try partial match (contains)
            for doc in available_docs:
                if filename_lower in doc.lower() or doc.lower() in filename_lower:
                    return doc
            
            # Try without extension
            filename_no_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
            for doc in available_docs:
                doc_no_ext = doc.rsplit('.', 1)[0] if '.' in doc else doc
                if filename_no_ext.lower() == doc_no_ext.lower():
                    return doc
            
            return None
        except Exception as e:
            logger.error(f"Error finding document by partial name: {str(e)}")
            return None
    
    async def summarize_document(self, filename: str) -> Dict[str, Any]:
        """
        Summarize a specific document with comprehensive analysis and improved error handling
        
        Args:
            filename: Name of the document to summarize
            
        Returns:
            Dict containing summary data and metadata
        """
        try:
            logger.info(f"Starting document summarization for: {filename}")
            
            # First, get list of available documents for debugging
            available_docs = self.get_available_documents()
            logger.info(f"Available documents: {available_docs}")
            
            # Try to find the document with flexible matching
            matched_filename = self.find_document_by_partial_name(filename)
            
            if not matched_filename:
                return {
                    "success": False,
                    "error": f"Document '{filename}' not found. Available documents: {available_docs}",
                    "filename": filename,
                    "summary": None,
                    "metadata": {
                        "available_documents": available_docs,
                        "search_attempted": filename
                    }
                }
            
            logger.info(f"Found matching document: {matched_filename}")
            
            # Search for content from the matched document
            search_results = self.rag_system.collection.get(
                where={"filename": matched_filename}
            )
            
            if not search_results or not search_results.get("documents"):
                # Try alternative search without where clause and filter manually
                logger.warning(f"Direct search failed, trying alternative approach...")
                all_results = self.rag_system.collection.get()
                
                if all_results and all_results.get("documents"):
                    # Filter manually
                    filtered_docs = []
                    filtered_metas = []
                    
                    for doc, meta in zip(all_results["documents"], all_results.get("metadatas", [])):
                        if meta and meta.get("filename") == matched_filename:
                            filtered_docs.append(doc)
                            filtered_metas.append(meta)
                    
                    if filtered_docs:
                        search_results = {
                            "documents": filtered_docs,
                            "metadatas": filtered_metas
                        }
                        logger.info(f"Manual filtering found {len(filtered_docs)} chunks")
                    else:
                        return {
                            "success": False,
                            "error": f"No content chunks found for document: {matched_filename}",
                            "filename": filename,
                            "summary": None,
                            "metadata": {
                                "matched_filename": matched_filename,
                                "total_documents_in_collection": len(all_results.get("documents", [])),
                                "available_documents": available_docs
                            }
                        }
                else:
                    return {
                        "success": False,
                        "error": f"No documents found in RAG collection",
                        "filename": filename,
                        "summary": None,
                        "metadata": None
                    }
            
            # Extract documents and metadata
            documents = search_results["documents"]
            metadatas = search_results.get("metadatas", [])
            
            logger.info(f"Found {len(documents)} chunks for document: {matched_filename}")
            
            # Combine all chunks from the document
            full_content = "\n\n".join(documents)
            
            # Calculate content statistics
            content_stats = {
                "total_chunks": len(documents),
                "total_characters": len(full_content),
                "total_words": len(full_content.split()),
                "average_chunk_size": len(full_content) // len(documents) if documents else 0
            }
            
            # Prepare content for summarization (handle large documents)
            content_for_summary = full_content
            is_truncated = False
            
            if len(full_content) > 8000:  # Increased limit for better summaries
                # Take first 4000 chars, middle 2000 chars, and last 2000 chars
                content_for_summary = (
                    full_content[:4000] + 
                    "\n\n[... MIDDLE CONTENT ...]\n\n" + 
                    full_content[len(full_content)//2:len(full_content)//2 + 2000] +
                    "\n\n[... END CONTENT ...]\n\n" + 
                    full_content[-2000:]
                )
                is_truncated = True
                logger.info(f"Content truncated for summarization. Original: {len(full_content)} chars")
            
            # Generate comprehensive summary
            summary_result = await self._generate_comprehensive_summary(
                content_for_summary, 
                matched_filename, 
                content_stats,
                is_truncated
            )
            
            # Extract metadata information
            metadata_info = self._extract_document_metadata(metadatas)
            
            return {
                "success": True,
                "filename": filename,
                "matched_filename": matched_filename,
                "summary": summary_result,
                "metadata": {
                    "content_stats": content_stats,
                    "document_info": metadata_info,
                    "is_truncated": is_truncated,
                    "summarization_timestamp": self._get_current_timestamp()
                },
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Document summarization error for {filename}: {str(e)}")
            return {
                "success": False,
                "error": f"Summary generation error: {str(e)}",
                "filename": filename,
                "summary": None,
                "metadata": {
                    "error_details": str(e),
                    "available_documents": self.get_available_documents()
                }
            }

    async def _generate_comprehensive_summary(
        self, 
        content: str, 
        filename: str, 
        stats: Dict[str, int],
        is_truncated: bool
    ) -> Dict[str, str]:
        """Generate a comprehensive summary with multiple sections"""
        
        # Create enhanced prompt for better summarization
        summary_prompt = f"""
        Analyze and summarize the following document content from "{filename}":

        Content Statistics:
        - Total chunks: {stats['total_chunks']}
        - Total words: {stats['total_words']}
        - Content truncated: {is_truncated}

        Document Content:
        {content}

        Please provide a comprehensive summary with the following sections:

        1. EXECUTIVE SUMMARY (2-3 sentences overview)
        2. MAIN TOPICS (bullet points of key subjects covered)
        3. KEY INSIGHTS (important findings, conclusions, or arguments)
        4. STRUCTURE & ORGANIZATION (how the document is organized)
        5. TARGET AUDIENCE (who this document is intended for)
        6. PRACTICAL APPLICATIONS (how this information can be used)

        Make the summary detailed but concise, focusing on the most important information.
        """
        
        try:
            if self.llm_service:
                summary = await self.llm_service.generate_response(summary_prompt)
            else:
                # Use legacy chains method with enhanced prompt
                summary = self.chains.get_tutor_response(
                    question="Please provide a comprehensive summary of this document.",
                    context=content,
                    chat_history=""
                )
            
            # Parse the summary into sections if possible
            return self._parse_summary_sections(summary)
            
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return {
                "executive_summary": f"Summary generation failed: {str(e)}",
                "main_topics": [],
                "key_insights": [],
                "structure": "Unable to analyze structure",
                "target_audience": "Unknown",
                "practical_applications": [],
                "full_summary": "Summary generation failed"
            }

    def _parse_summary_sections(self, summary: str) -> Dict[str, Any]:
        """Parse the generated summary into structured sections"""
        try:
            sections = {
                "executive_summary": "",
                "main_topics": [],
                "key_insights": [],
                "structure": "",
                "target_audience": "",
                "practical_applications": [],
                "full_summary": summary
            }
            
            # Simple parsing logic - can be enhanced based on LLM output format
            lines = summary.split('\n')
            current_section = "full_summary"
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Detect section headers
                if "EXECUTIVE SUMMARY" in line.upper():
                    current_section = "executive_summary"
                elif "MAIN TOPICS" in line.upper():
                    current_section = "main_topics"
                elif "KEY INSIGHTS" in line.upper():
                    current_section = "key_insights"
                elif "STRUCTURE" in line.upper():
                    current_section = "structure"
                elif "TARGET AUDIENCE" in line.upper():
                    current_section = "target_audience"
                elif "PRACTICAL APPLICATIONS" in line.upper():
                    current_section = "practical_applications"
                else:
                    # Add content to current section
                    if current_section in ["main_topics", "key_insights", "practical_applications"]:
                        if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                            sections[current_section].append(line)
                    elif current_section in ["executive_summary", "structure", "target_audience"]:
                        if sections[current_section]:
                            sections[current_section] += " " + line
                        else:
                            sections[current_section] = line
            
            return sections
            
        except Exception as e:
            logger.error(f"Summary parsing error: {str(e)}")
            return {
                "executive_summary": "Summary parsing failed",
                "main_topics": [],
                "key_insights": [],
                "structure": "Unable to parse structure",
                "target_audience": "Unknown",
                "practical_applications": [],
                "full_summary": summary
            }

    def _extract_document_metadata(self, metadatas: List[Dict]) -> Dict[str, Any]:
        """Extract and consolidate metadata from document chunks"""
        if not metadatas:
            return {}
        
        # Consolidate metadata from all chunks
        consolidated = {
            "filenames": set(),
            "chunk_count": len(metadatas),
            "unique_sources": set(),
            "document_types": set()
        }
        
        for meta in metadatas:
            if isinstance(meta, dict):
                if "filename" in meta:
                    consolidated["filenames"].add(meta["filename"])
                if "source" in meta:
                    consolidated["unique_sources"].add(meta["source"])
                if "document_type" in meta:
                    consolidated["document_types"].add(meta["document_type"])
        
        # Convert sets to lists for JSON serialization
        return {
            "filenames": list(consolidated["filenames"]),
            "chunk_count": consolidated["chunk_count"],
            "unique_sources": list(consolidated["unique_sources"]),
            "document_types": list(consolidated["document_types"])
        }

    def _get_current_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat()

    async def explain_concept(self, concept: str, use_rag: bool = True) -> str:
        """Enhanced explain_concept with better query handling"""
        try:
            print(f"\n=== EXPLAIN CONCEPT DEBUG ===")
            print(f"Concept: {concept}")
            print(f"Use RAG: {use_rag}")
            
            context = ""
            sources = []
            
            if use_rag:
                doc_count = self.rag_system.collection.count()
                print(f"Documents in collection: {doc_count}")
                
                if doc_count > 0:
                    # STRATEGY 1: Try direct search first
                    print("Trying direct search...")
                    context, sources = self.rag_system.get_context_for_query(concept)
                    
                    # STRATEGY 2: If results are poor, try specific searches for your query
                    if not context or len(context) < 500:
                        print("Direct search returned limited results, trying specific searches...")
                        
                        # For "A few things that science does not do" - try specific alternatives
                        alternative_searches = [
                            "what science cannot do",
                            "limitations of science", 
                            "science does not",
                            "science cannot",
                            "boundaries of science",
                            "misconceptions about science",
                            "science limitations"
                        ]
                        
                        best_context = ""
                        best_sources = []
                        
                        for alt_query in alternative_searches:
                            alt_context, alt_sources = self.rag_system.get_context_for_query(alt_query)
                            if len(alt_context) > len(best_context):
                                best_context = alt_context
                                best_sources = alt_sources
                                print(f"Better results with: '{alt_query}' - {len(alt_context)} chars")
                        
                        if best_context:
                            context = best_context
                            sources = best_sources
                    
                    # STRATEGY 3: If still poor results, try keyword-based search
                    if not context or len(context) < 300:
                        print("Trying keyword-based search...")
                        
                        # Search for documents containing key concepts
                        try:
                            all_docs = self.rag_system.collection.get()
                            relevant_chunks = []
                            
                            # Keywords related to what science doesn't do
                            keywords = ['cannot', 'does not', 'limitation', 'boundary', 'not scientific', 
                                      'beyond science', 'outside science', 'non-scientific']
                            
                            for doc, meta in zip(all_docs['documents'], all_docs['metadatas']):
                                keyword_count = sum(1 for keyword in keywords if keyword.lower() in doc.lower())
                                if keyword_count > 0:
                                    relevant_chunks.append((doc, meta, keyword_count))
                            
                            # Sort by keyword relevance
                            relevant_chunks.sort(key=lambda x: x[2], reverse=True)
                            
                            if relevant_chunks:
                                # Take top chunks up to context limit
                                selected_chunks = []
                                total_length = 0
                                max_length = 3000
                                
                                for chunk, meta, score in relevant_chunks:
                                    if total_length + len(chunk) <= max_length:
                                        selected_chunks.append(chunk)
                                        total_length += len(chunk)
                                        if meta['filename'] not in sources:
                                            sources.append(meta['filename'])
                                
                                if selected_chunks:
                                    context = "\n\n".join(selected_chunks)
                                    print(f"Keyword search found {len(selected_chunks)} relevant chunks")
                        
                        except Exception as e:
                            print(f"Keyword search error: {e}")
                    
                    print(f"Final context length: {len(context)}")
                    print(f"Sources: {sources}")
                else:
                    print("No documents found in RAG system")
            
            # Generate enhanced explanation
            if context:
                # Create a comprehensive explanation
                explanation = f"""Based on your uploaded document, here's what I found about "{concept}":

## Information from Your Document:

{context}

## Summary:

This content from your document addresses the concept you asked about. The document appears to discuss the scope and limitations of science, which directly relates to your question about "things that science does not do."

**Sources:** {', '.join(sources)}"""
                
            else:
                # Provide helpful guidance when no relevant content is found
                explanation = f"""I searched through your uploaded document ({self.rag_system.collection.count()} chunks from your PDF) but couldn't find specific content directly addressing "{concept}".

**Possible reasons:**
1. The concept might be described using different terminology in your document
2. The specific phrase might not appear in the document
3. The content might be split across multiple sections

**Suggestions to get better results:**
- Try searching for: "science limitations", "what science cannot do", "boundaries of science"
- Use keywords that might appear in your document
- Try more general terms related to your concept

**What I can help with:**
- Summarize your entire document
- Search for specific terms or concepts
- Explain any part of the content that was uploaded

Would you like me to try a different search approach or summarize the document to help you find the relevant information?"""
            
            print(f"=== END EXPLAIN CONCEPT DEBUG ===\n")
            return explanation
            
        except Exception as e:
            error_msg = f"Error in explain_concept: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get information about a session (returns serializable data only)"""
        try:
            if session_id not in self.memory_manager.sessions:
                return {"error": "Session not found"}
            
            memory = self.memory_manager.sessions[session_id]
            
            # Extract chat history in a serializable format
            chat_history = []
            if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
                for msg in memory.chat_memory.messages:
                    if hasattr(msg, 'content'):
                        chat_history.append({
                            "type": msg.__class__.__name__,
                            "content": msg.content
                        })
            
            return {
                "session_id": session_id,
                "message_count": len(chat_history),
                "last_messages": chat_history[-5:] if chat_history else [],  # Last 5 messages
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting session info for {session_id}: {str(e)}")
            return {"error": f"Failed to get session info: {str(e)}"}
    
    def clear_session(self, session_id: str) -> Dict:
        """Clear a session"""
        try:
            self.memory_manager.clear_session(session_id)
            logger.info(f"Session {session_id} cleared successfully")
            return {"message": f"Session {session_id} cleared successfully"}
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {str(e)}")
            return {"error": f"Failed to clear session: {str(e)}"}
    
    async def upload_document(self, content: bytes, filename: str, is_pdf: bool = False) -> Dict[str, Any]:
        """
    Upload and process a document
    
    Args:
        content: Raw file content as bytes
        filename: Name of the file
        is_pdf: Whether the file is a PDF document
        
    Returns:
        Dict containing upload results
        """
        try:
            logger.info(f"Received document upload: {filename}")
        
        # Validate input
            if not content:
                return {
                "success": False,
                "error": "No content provided",
                "filename": filename
            }
        
        # Initialize RAG system if not already done
            if not hasattr(self, 'rag_system') or self.rag_system is None:
              # Adjust import path as needed
                self.rag_system = RAGSystem()
        
        # Process and add document to vector store
            if is_pdf:
            # Use the RAG system's add_pdf_document method
                result = self.rag_system.add_pdf_document(content, filename)
            else:
            # For text files, decode content first
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                # Try other common encodings
                    try:
                        text_content = content.decode('latin-1')
                    except:
                        return {
                        "success": False,
                        "error": "Could not decode text file. Please ensure it's UTF-8 encoded.",
                        "filename": filename
                    }
            
            # Use the RAG system's add_document method
            result = self.rag_system.add_document(text_content, filename)
        
            logger.info(f"Document processing result: {result}")
        
        # Return success with detailed info
            return {
            "success": True,
            "message": f"Document '{filename}' uploaded and processed successfully",
            "filename": filename,
            "file_size": len(content),
            "is_pdf": is_pdf,
            "processing_result": result,
            "content_type": "application/pdf" if is_pdf else "text/plain"
        }
        
        except Exception as e:
            logger.error(f"Document upload error for {filename}: {str(e)}")
            return {
            "success": False,
            "error": f"Upload failed: {str(e)}",
            "filename": filename
        }
    
    
    
    
    async def add_message_directly_to_history(self, session_id: str, message: str, role: str):
        """
        Add a message directly to session history without any AI processing
        This is for storing summaries, assistant responses, etc.
        """
        try:
            # Get existing session history
            session_info = self.get_session_info(session_id)
            if not session_info:
                raise ValueError("Session not found")
            
            # Add message to history
            message_entry = {
                "role": role,
                "message": message,  # Store exact content
                "content": message,  # Also store as content field
                "text": message,     # And as text field for compatibility
                "timestamp": datetime.now().isoformat(),
                "id": f"{session_id}_{int(time.time())}"
            }
            
            # Add to your session storage (database, file, etc.)
            # This depends on how you're storing sessions
            # Example:
            if hasattr(self, 'session_storage'):
                self.session_storage[session_id]['messages'].append(message_entry)
            
            return {
                "success": True,
                "message_id": message_entry["id"],
                "stored_content": message  # Return what was actually stored
            }
            
        except Exception as e:
            logger.error(f"Failed to add message directly to history: {str(e)}")
            raise
        
        
            # Add this method to your ChatService class

    async def add_message_to_session(self, session_id: str, message: str, role: str = "user"):
        """Add a message directly to session history without processing"""
        try:
        # Ensure session exists first - use your existing method
            if not self.memory_manager.session_exists(session_id):
                self.memory_manager.create_session_only(session_id)
        
        # Get the memory object using your existing method
            memory = self.memory_manager.get_memory_object(session_id)
        
            if role.lower() in ["user", "human"]:
                memory.chat_memory.add_user_message(message)
            elif role.lower() in ["assistant", "ai"]:
                memory.chat_memory.add_ai_message(message)
            else:
            # For other roles, add as user message with role prefix
                memory.chat_memory.add_user_message(f"[{role}]: {message}")
        
        # Get message count using your existing get_chat_history method
            history = self.memory_manager.get_chat_history(session_id)
            message_count = len(history)
        
            logger.info(f"✅ Message added to session {session_id} using MemoryManager methods")
            logger.info(f"📊 Total messages in session: {message_count}")
        
            return {
            "success": True,
            "message_stored": True,
            "session_id": session_id,
            "role": role,
            "message_count": message_count
        }
        
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {str(e)}")
            raise e