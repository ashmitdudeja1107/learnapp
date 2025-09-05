from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from typing import Dict, List
import json
import logging

# Import your LLM service
from services.llm_service import create_llm_service, LLMService  # Adjust import path as needed

logger = logging.getLogger(__name__)

class TutorChains:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        # Initialize with Groq LLM service using the same model as LLMService
        self.llm_service = create_llm_service(
            provider="groq",
            model_name=model_name,
            temperature=0.7,
            max_tokens=2000
        )
        
        # AI Tutor prompt template
        self.tutor_prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""You are an AI tutor designed to help students learn effectively. You are knowledgeable, patient, and encouraging.

Context from documents:
{context}

Chat History:
{chat_history}

Student Question: {question}

Instructions:
1. If the context contains relevant information, use it to provide accurate answers
2. Break down complex concepts into simple, understandable parts
3. Provide examples when helpful
4. Ask follow-up questions to ensure understanding
5. Encourage the student and provide positive reinforcement
6. If you don't know something, admit it and suggest how the student might find the answer
7. Adapt your explanation style to the student's level of understanding

Response:"""
        )
        
        # Quiz generation prompt
        self.quiz_prompt = PromptTemplate(
            input_variables=["context", "topic"],
            template="""Based on the following context, create a quiz about {topic}.

Context:
{context}

Create 5 multiple choice questions with 4 options each (A, B, C, D). 
Format your response as JSON with the following structure:
{{
    "questions": [
        {{
            "question": "Question text here",
            "options": {{
                "A": "Option A",
                "B": "Option B", 
                "C": "Option C",
                "D": "Option D"
            }},
            "correct_answer": "A",
            "explanation": "Explanation of why this is correct"
        }}
    ]
}}

Quiz:"""
        )
        
        # Summary prompt
        self.summary_prompt = PromptTemplate(
            input_variables=["context"],
            template="""Summarize the following content in a clear, educational manner. 
Focus on the key concepts and learning points:

{context}

Summary:"""
        )
    
    def get_tutor_response(self, question: str, context: str = "", chat_history: str = "") -> str:
        """Generate tutor response using context and chat history"""
        prompt = self.tutor_prompt.format(
            context=context,
            chat_history=chat_history,
            question=question
        )
        
        try:
            response = self.llm_service.generate_text(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating tutor response: {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please try again."
    
    def generate_quiz(self, topic: str, context: str = "") -> str:
        """Generate a quiz based on topic and context"""
        prompt = self.quiz_prompt.format(
            topic=topic,
            context=context
        )
        
        try:
            response = self.llm_service.generate_text(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating quiz: {str(e)}")
            return json.dumps({"error": "Failed to generate quiz", "details": str(e)})
    
    def summarize_content(self, context: str) -> str:
        """Summarize given content"""
        prompt = self.summary_prompt.format(context=context)
        
        try:
            response = self.llm_service.generate_text(prompt)
            return response
        except Exception as e:
            logger.error(f"Error summarizing content: {str(e)}")
            return "I apologize, but I encountered an error while summarizing the content. Please try again."
    
    def explain_concept(self, concept: str, context: str = "", sources: List[str] = None) -> str:
        """
        Explain a specific concept using RAG context when available
        """
        if context and context.strip():
            # We have relevant context from uploaded documents
            explanation_prompt = f"""You are an AI tutor. Explain the concept of "{concept}" using the provided context from uploaded documents.

CONTEXT FROM UPLOADED DOCUMENTS:
{context}

SOURCES: {', '.join(sources) if sources else 'Not specified'}

Based on the context above, please provide:
1. A clear definition of "{concept}"
2. Key characteristics or components (from the documents)
3. Real-world examples (preferably from the context)
4. How it relates to other concepts mentioned in the documents
5. Any specific details or insights from the uploaded materials

Important: Base your explanation primarily on the provided context. If the context doesn't fully cover the concept, mention what information is available from the documents and supplement with general knowledge where needed.

Explanation:"""
        else:
            # No context available - fall back to general explanation
            explanation_prompt = f"""You are an AI tutor. Explain the concept of "{concept}" in a clear, educational way.

Note: No specific context from uploaded documents is available for this concept.

Please provide:
1. A clear definition
2. Key characteristics or components
3. Real-world examples
4. How it relates to other concepts
5. Common misconceptions (if any)

Explanation:"""
        
        try:
            response = self.llm_service.generate_text(explanation_prompt)
            return response
        except Exception as e:
            logger.error(f"Error explaining concept: {str(e)}")
            return f"I apologize, but I encountered an error while explaining the concept '{concept}'. Please try again."