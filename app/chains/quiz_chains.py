from langchain.schema import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from typing import List, Dict, Any, Optional
import json
import re
import logging
from database.models.quiz_models import QuizQuestion, QuizOption, DifficultyLevel

logger = logging.getLogger(__name__)

class QuizQuestionParser(BaseOutputParser):
    """Custom parser for quiz question responses"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM response into structured quiz question format"""
        try:
            # Clean the text first
            text = text.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # Try to parse as direct JSON
            if text.startswith('{') and text.endswith('}'):
                return json.loads(text)
            
            # Fallback to manual parsing
            return self._manual_parse(text)
            
        except Exception as e:
            logger.error(f"Error parsing quiz question: {str(e)}")
            return self._create_fallback_question(text)
    
    def _manual_parse(self, text: str) -> Dict[str, Any]:
        """Manual parsing when JSON parsing fails"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        question = ""
        options = []
        correct_answer = "A"
        explanation = ""
        
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            if "question:" in line_lower:
                question = line.split(":", 1)[1].strip()
                current_section = "question"
            elif "options:" in line_lower:
                current_section = "options"
            elif "correct:" in line_lower or "answer:" in line_lower:
                # Extract correct answer
                answer_match = re.search(r'[ABCD]', line)
                if answer_match:
                    correct_answer = answer_match.group()
                current_section = "answer"
            elif "explanation:" in line_lower:
                explanation = line.split(":", 1)[1].strip()
                current_section = "explanation"
            elif current_section == "options" and re.match(r'^[ABCD]\)', line):
                options.append(line)
            elif current_section == "explanation" and not any(keyword in line_lower for keyword in ["question:", "options:", "correct:", "answer:"]):
                explanation += " " + line
        
        # Ensure we have 4 options
        if len(options) < 4:
            default_options = ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"]
            options = options + default_options[len(options):]
        
        return {
            "question": question or "Generated question from content",
            "options": options[:4],
            "correct_answer": correct_answer,
            "explanation": explanation or "Based on the provided content"
        }
    
    def _create_fallback_question(self, text: str) -> Dict[str, Any]:
        """Create a fallback question when parsing completely fails"""
        return {
            "question": "What is the main topic discussed in the provided content?",
            "options": ["A) Topic A", "B) Topic B", "C) Topic C", "D) Topic D"],
            "correct_answer": "A",
            "explanation": "This is a fallback question generated when parsing failed."
        }

class QuizChains:
    """Modern LangChain chains for quiz generation using LCEL"""
    
    def __init__(self, llm):
        self.llm = llm
        self.question_parser = QuizQuestionParser()
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup all the chains for quiz generation using LCEL"""
        self.multiple_choice_chain = self._create_multiple_choice_chain()
        self.true_false_chain = self._create_true_false_chain()
        self.difficulty_analyzer_chain = self._create_difficulty_analyzer_chain()
        self.topic_extractor_chain = self._create_topic_extractor_chain()
    
    def _create_multiple_choice_chain(self) -> RunnableSequence:
        """Create chain for multiple choice questions using LCEL"""
        template = """
Based on the following content, create a {difficulty} level multiple choice question.

Content: {content}

Requirements:
- Question should test understanding of the key concepts
- Provide exactly 4 options (A, B, C, D)
- Only one option should be correct
- Make incorrect options plausible but clearly wrong
- Provide a clear explanation for the correct answer

Format your response as JSON:
{{
    "question": "Your question here",
    "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],
    "correct_answer": "A",
    "explanation": "Explanation of why this answer is correct"
}}

Question:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["content", "difficulty"]
        )
        
        return prompt | self.llm | self.question_parser
    
    def _create_true_false_chain(self) -> RunnableSequence:
        """Create chain for true/false questions using LCEL"""
        template = """
Based on the following content, create a {difficulty} level true/false question.

Content: {content}

Requirements:
- Question should be clearly true or false based on the content
- Avoid ambiguous statements
- Provide a clear explanation

Format your response as JSON:
{{
    "question": "Your true/false statement here",
    "options": ["A) True", "B) False"],
    "correct_answer": "A",
    "explanation": "Explanation of why this is true/false"
}}

Question:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["content", "difficulty"]
        )
        
        return prompt | self.llm | self.question_parser
    
    def _create_difficulty_analyzer_chain(self) -> RunnableSequence:
        """Create chain to analyze content difficulty using LCEL"""
        template = """
Analyze the following content and determine its complexity level.

Content: {content}

Consider:
- Vocabulary complexity
- Concept difficulty
- Technical depth
- Required background knowledge

Respond with one word: "easy", "medium", or "hard"

Difficulty:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["content"]
        )
        
        return prompt | self.llm
    
    def _create_topic_extractor_chain(self) -> RunnableSequence:
        """Create chain to extract main topics from content using LCEL"""
        template = """
Extract the main topics/themes from the following content.

Content: {content}

List the top 3 main topics, separated by commas.
Focus on the most important concepts or subjects discussed.

Topics:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["content"]
        )
        
        return prompt | self.llm
    
    def generate_multiple_choice_question(self, content: str, difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a multiple choice question from content"""
        try:
            result = self.multiple_choice_chain.invoke({
                "content": content[:1500],  # Limit content length
                "difficulty": difficulty
            })
            return result
        except Exception as e:
            logger.error(f"Error generating multiple choice question: {str(e)}")
            return self._create_error_question()
    
    def generate_true_false_question(self, content: str, difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a true/false question from content"""
        try:
            result = self.true_false_chain.invoke({
                "content": content[:1500],
                "difficulty": difficulty
            })
            return result
        except Exception as e:
            logger.error(f"Error generating true/false question: {str(e)}")
            return self._create_error_question(question_type="true_false")
    
    def analyze_content_difficulty(self, content: str) -> str:
        """Analyze the difficulty level of content"""
        try:
            result = self.difficulty_analyzer_chain.invoke({"content": content[:1000]})
            
            # Handle different response types
            if hasattr(result, 'content'):
                difficulty = result.content.strip().lower()
            else:
                difficulty = str(result).strip().lower()
            
            if difficulty in ["easy", "medium", "hard"]:
                return difficulty
            else:
                return "medium"  # Default fallback
                
        except Exception as e:
            logger.error(f"Error analyzing content difficulty: {str(e)}")
            return "medium"
    
    def extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content"""
        try:
            result = self.topic_extractor_chain.invoke({"content": content[:1000]})
            
            # Handle different response types
            if hasattr(result, 'content'):
                topics_text = result.content.strip()
            else:
                topics_text = str(result).strip()
            
            topics = [topic.strip() for topic in topics_text.split(',')]
            return topics[:3]  # Return top 3 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return ["General Topic"]
    
    def _create_error_question(self, question_type: str = "multiple_choice") -> Dict[str, Any]:
        """Create a fallback question when generation fails"""
        if question_type == "true_false":
            return {
                "question": "The provided content contains useful information.",
                "options": ["A) True", "B) False"],
                "correct_answer": "A",
                "explanation": "This is a fallback question due to generation error."
            }
        else:
            return {
                "question": "What type of content was provided?",
                "options": ["A) Text content", "B) No content", "C) Invalid content", "D) Unknown content"],
                "correct_answer": "A",
                "explanation": "This is a fallback question due to generation error."
            }

class QuizGenerationPipeline:
    """Complete pipeline for quiz generation"""
    
    def __init__(self, quiz_chains: QuizChains, quiz_rag_service):
        self.chains = quiz_chains
        self.rag_service = quiz_rag_service
    def generate_quiz_questions(self, 
                              content_chunks: List[str], 
                              num_questions: int = 8,
                              difficulty: str = "medium",
                              question_type: str = "multiple_choice") -> List[QuizQuestion]:
        """Generate a complete set of quiz questions"""
        questions = []
        
        # FIXED: The issue was here - it was limited by min(num_questions, len(content_chunks))
        # Now we cycle through content chunks if we need more questions than chunks available
        
        for i in range(num_questions):
            try:
                # Cycle through content chunks if we need more questions than available chunks
                content_index = i % len(content_chunks)
                content = content_chunks[content_index]
                
                # Add some variation by using different parts of the same content
                # if we're reusing content chunks
                if i >= len(content_chunks) and len(content) > 1000:
                    # Split large content into different sections for variety
                    section_size = len(content) // 3
                    section_start = (i % 3) * section_size
                    section_end = section_start + section_size + 200  # Add overlap
                    content = content[section_start:section_end]
                
                # Generate question based on type
                if question_type == "true_false":
                    question_data = self.chains.generate_true_false_question(content, difficulty)
                else:
                    question_data = self.chains.generate_multiple_choice_question(content, difficulty)
                
                # Extract topic from content
                topics = self.chains.extract_topics(content)
                topic = topics[0] if topics else "General"
                
                # Create QuizQuestion object
                options = []
                for opt_text in question_data["options"]:
                    # Handle different option formats
                    if isinstance(opt_text, str):
                        if len(opt_text) > 2 and opt_text[1] == ')':
                            label = opt_text[0]  # Extract A, B, C, D
                            text = opt_text[3:]  # Remove "A) " prefix
                        else:
                            # Fallback for malformed options
                            label = chr(65 + len(options))  # A, B, C, D
                            text = opt_text
                    else:
                        # Handle non-string options
                        label = chr(65 + len(options))
                        text = str(opt_text)
                    
                    is_correct = (label == question_data["correct_answer"])
                    
                    options.append(QuizOption(
                        label=label,
                        text=text,
                        is_correct=is_correct
                    ))
                
                quiz_question = QuizQuestion(
                    id=f"q_{i+1}",
                    question=question_data["question"],
                    options=options,
                    correct_answer=question_data["correct_answer"],
                    explanation=question_data["explanation"],
                    difficulty=DifficultyLevel(difficulty),
                    topic=topic,
                    context=content[:200] + "..." if len(content) > 200 else content
                )
                
                questions.append(quiz_question)
                
            except Exception as e:
                logger.error(f"Error generating question {i+1}: {str(e)}")
                # Don't continue on error - this might be why you're getting fewer questions
                # Instead, create a fallback question
                fallback_question = self._create_fallback_quiz_question(i+1, difficulty)
                if fallback_question:
                    questions.append(fallback_question)
        
        return questions

    def _validate_question_data(self, question_data: dict) -> bool:
        """Validate that question_data has required fields"""
        required_fields = ["question", "options", "correct_answer", "explanation"]
    
        for field in required_fields:
            if field not in question_data:
                logger.error(f"Missing required field: {field}")
                return False
    
        if not isinstance(question_data["options"], list) or len(question_data["options"]) < 2:
            logger.error("Invalid options: must be a list with at least 2 options")
            return False
    
        return True