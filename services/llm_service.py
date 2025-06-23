from typing import List, Dict, Any, Optional, Union
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

@dataclass
class LLMConfig:
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30

class LLMService:
    """Main LLM Service that creates LangChain-compatible LLM instances"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = self._create_langchain_model(config)
        logger.info(f"LLM Service initialized with {config.provider} provider")
    
    def _create_langchain_model(self, config: LLMConfig):
        """Create LangChain-compatible LLM model"""
        if config.provider == LLMProvider.OPENAI:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=config.model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    openai_api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
                    openai_api_base=config.api_base,
                    timeout=config.timeout
                )
            except ImportError:
                raise ImportError("langchain-openai package not installed. Install with: pip install langchain-openai")
        
        elif config.provider == LLMProvider.ANTHROPIC:
            try:
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model=config.model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    anthropic_api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
                    timeout=config.timeout
                )
            except ImportError:
                raise ImportError("langchain-anthropic package not installed. Install with: pip install langchain-anthropic")
        
        elif config.provider == LLMProvider.OLLAMA:
            try:
                # Fixed import - use the new langchain-ollama package instead of deprecated langchain_community.llms
                from langchain_ollama import OllamaLLM
                return OllamaLLM(
                    model=config.model_name,
                    base_url=config.api_base or "http://localhost:11434",
                    temperature=config.temperature,
                    num_predict=config.max_tokens
                )
            except ImportError:
                raise ImportError("langchain-ollama package not installed. Install with: pip install langchain-ollama")
        
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the configured model"""
        try:
            if hasattr(self.model, 'invoke'):
                # For chat models
                from langchain_core.messages import HumanMessage
                response = self.model.invoke([HumanMessage(content=prompt)])
                return response.content if hasattr(response, 'content') else str(response)
            else:
                # For regular LLMs
                response = self.model.invoke(prompt)
                return response
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    async def generate_text_async(self, prompt: str, **kwargs) -> str:
        """Generate text asynchronously"""
        try:
            if hasattr(self.model, 'ainvoke'):
                if hasattr(self.model, 'invoke') and 'chat' in str(type(self.model)).lower():
                    # For chat models
                    from langchain_core.messages import HumanMessage
                    response = await self.model.ainvoke([HumanMessage(content=prompt)])
                    return response.content if hasattr(response, 'content') else str(response)
                else:
                    # For regular LLMs
                    response = await self.model.ainvoke(prompt)
                    return response
            else:
                # Fallback to sync version
                return await asyncio.to_thread(self.generate_text, prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error generating text async: {str(e)}")
            raise
    
    def generate_quiz_questions(self, 
                              content: str, 
                              num_questions: int = 5, 
                              difficulty: str = "medium",
                              question_type: str = "multiple_choice") -> List[Dict[str, Any]]:
        """Generate quiz questions from content"""
        
        prompt = f"""
        Based on the following content, generate {num_questions} {difficulty} difficulty {question_type} questions.
        
        Content:
        {content}
        
        Requirements:
        - Generate exactly {num_questions} questions
        - Difficulty level: {difficulty}
        - Question type: {question_type}
        - Each question should have 4 options (A, B, C, D)
        - Include explanations for correct answers
        - Questions should test understanding, not just memorization
        - Vary the topics covered in the content
        
        Please respond with a JSON array of questions in this format:
        [
            {{
                "id": "q1",
                "question": "Question text here?",
                "options": [
                    {{"label": "A", "text": "Option A text", "is_correct": false}},
                    {{"label": "B", "text": "Option B text", "is_correct": true}},
                    {{"label": "C", "text": "Option C text", "is_correct": false}},
                    {{"label": "D", "text": "Option D text", "is_correct": false}}
                ],
                "correct_answer": "B",
                "explanation": "Explanation for why B is correct",
                "difficulty": "{difficulty}",
                "topic": "Topic name"
            }}
        ]
        """
        
        try:
            response = self.generate_text(prompt)
            
            # Try to parse JSON response
            try:
                questions = json.loads(response)
                return questions
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    questions = json.loads(json_match.group())
                    return questions
                else:
                    logger.error("Could not extract valid JSON from quiz generation response")
                    return []
                    
        except Exception as e:
            logger.error(f"Error generating quiz questions: {str(e)}")
            return []
    
    def evaluate_answer_explanation(self, 
                                  question: str, 
                                  correct_answer: str, 
                                  user_answer: str, 
                                  context: str = "") -> str:
        """Generate detailed explanation for quiz answer"""
        
        prompt = f"""
        Provide a detailed explanation for this quiz question:
        
        Question: {question}
        Correct Answer: {correct_answer}
        User's Answer: {user_answer}
        Context: {context}
        
        Please explain:
        1. Why the correct answer is right
        2. If the user's answer is wrong, explain why
        3. Key concepts involved
        4. Any additional insights
        
        Keep the explanation clear and educational.
        """
        
        return self.generate_text(prompt)
    
    def generate_question_from_context(self, 
                                     context: str, 
                                     difficulty: str = "medium",
                                     question_type: str = "multiple_choice") -> Optional[Dict[str, Any]]:
        """Generate a single question from given context"""
        
        prompt = f"""
        Based on this specific context, generate 1 {difficulty} difficulty {question_type} question.
        
        Context:
        {context}
        
        Generate a question that tests understanding of this specific content.
        
        Respond with JSON in this format:
        {{
            "id": "q1",
            "question": "Question text here?",
            "options": [
                {{"label": "A", "text": "Option A text", "is_correct": false}},
                {{"label": "B", "text": "Option B text", "is_correct": true}},
                {{"label": "C", "text": "Option C text", "is_correct": false}},
                {{"label": "D", "text": "Option D text", "is_correct": false}}
            ],
            "correct_answer": "B",
            "explanation": "Explanation for why B is correct",
            "difficulty": "{difficulty}",
            "topic": "Topic name"
        }}
        """
        
        try:
            response = self.generate_text(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error generating single question: {str(e)}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check if LLM service is working"""
        try:
            test_response = self.generate_text("Hello, respond with 'OK' if you can see this.")
            return {
                "status": "healthy",
                "provider": self.config.provider,
                "model": self.config.model_name,
                "test_response": test_response[:50] + "..." if len(test_response) > 50 else test_response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.config.provider,
                "model": self.config.model_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Factory function to create LLM service with different providers
def create_llm_service(provider: str = "openai", 
                      model_name: str = "gpt-3.5-turbo",
                      api_key: Optional[str] = None,
                      **kwargs) -> LLMService:
    """Factory function to create LLM service"""
    
    config = LLMConfig(
        provider=LLMProvider(provider),
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )
    
    return LLMService(config)

# Example usage and configuration
def get_default_openai_service() -> LLMService:
    """Get default OpenAI service configuration"""
    return create_llm_service(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2000
    )

def get_default_anthropic_service() -> LLMService:
    """Get default Anthropic service configuration"""
    return create_llm_service(
        provider="anthropic",
        model_name="claude-3-sonnet-20240229",
        temperature=0.7,
        max_tokens=2000
    )

def get_default_ollama_service(model_name: str = "llama2") -> LLMService:
    """Get default Ollama service configuration"""
    return create_llm_service(
        provider="ollama",
        model_name=model_name,
        api_base="http://localhost:11434",
        temperature=0.7,
        max_tokens=2000
    )