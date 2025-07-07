"""
Query Router for Co-Researcher.AI
=================================

This module determines whether a user query should be answered directly
or needs a full research pipeline. It uses LLM classification to make
intelligent routing decisions.

Author: Co-Researcher.AI Team
"""

import logging
from typing import Literal, Dict, List, Any

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Routes queries to appropriate handlers based on complexity and intent
    
    This router analyzes user queries and determines whether they need:
    - Direct answer (general knowledge questions)
    - Research pipeline (literature review questions)
    - Keyword extraction and optimization for academic search
    """
    
    def __init__(self, llm_client):
        """
        Initialize the router with LLM client
        
        Args:
            llm_client: Instance of LLMClient for query classification
        """
        self.llm_client = llm_client
        logger.info("🧭 Query Router initialized")
    
    def classify_query(self, query: str) -> Literal["general", "research"]:
        """
        Classify a query as either general or research-focused
        
        Args:
            query: User's input question
            
        Returns:
            "general" for direct questions, "research" for literature review needs
        """
        logger.info(f"🤔 Classifying query: {query[:50]}...")
        
        try:
            # Use the LLMClient's classify_query method
            classification = self.llm_client.classify_query(query)
            
            logger.info(f"✅ Query classified as: {classification}")
            return classification
            
        except Exception as e:
            logger.error(f"❌ Error classifying query: {e}")
            # Default to research on error to be safe
            return "research"
    
    def extract_academic_keywords(self, query: str) -> Dict[str, Any]:
        """
        Extract and optimize keywords from natural language queries for academic search
        
        This method transforms user queries like "I want to know recent findings about LLMs" 
        into optimized search terms like "large language models", "transformer architectures", 
        "recent developments", etc.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary with optimized keywords and search strategy
        """
        logger.info(f"🔍 Extracting academic keywords from: {query[:50]}...")
        
        keyword_extraction_prompt = f"""You are an expert academic search strategist. Transform the user's natural language query into optimized search terms for academic databases like arXiv and PubMed.

User Query: "{query}"

Analyze the query and provide a comprehensive search strategy with the following JSON structure:

{{
    "primary_keywords": ["core technical terms", "main concepts"],
    "secondary_keywords": ["related terms", "synonyms", "alternative phrases"],
    "domain_specific_terms": ["field-specific terminology", "technical jargon"],
    "temporal_indicators": ["recent", "latest", "current", "2024", "state-of-the-art"],
    "methodology_keywords": ["algorithm", "approach", "method", "technique"],
    "comparison_terms": ["vs", "comparison", "versus", "different approaches"],
    "optimized_query": "Best single search query for academic databases",
    "search_strategy": "brief explanation of search approach",
    "suggested_sources": ["arXiv", "PubMed", "both"],
    "domain": "primary research field"
}}

EXAMPLES:

Query: "I want to know recent findings about LLMs"
Response: {{
    "primary_keywords": ["large language models", "LLM", "transformer models"],
    "secondary_keywords": ["neural language models", "generative models", "language AI"],
    "domain_specific_terms": ["GPT", "BERT", "transformer architecture", "attention mechanism"],
    "temporal_indicators": ["recent", "latest", "2024", "current", "state-of-the-art"],
    "methodology_keywords": ["training", "fine-tuning", "pre-training", "evaluation"],
    "comparison_terms": [],
    "optimized_query": "large language models recent developments transformer architecture",
    "search_strategy": "Focus on recent LLM papers with broad transformer coverage",
    "suggested_sources": ["arXiv"],
    "domain": "Natural Language Processing"
}}

Query: "Compare federated learning privacy approaches"
Response: {{
    "primary_keywords": ["federated learning", "privacy preservation", "privacy-preserving"],
    "secondary_keywords": ["distributed learning", "collaborative learning", "decentralized ML"],
    "domain_specific_terms": ["differential privacy", "secure aggregation", "homomorphic encryption"],
    "temporal_indicators": ["recent", "current"],
    "methodology_keywords": ["approach", "method", "technique", "algorithm"],
    "comparison_terms": ["comparison", "comparative", "versus", "different approaches"],
    "optimized_query": "federated learning privacy preservation comparison differential privacy",
    "search_strategy": "Search for comparative studies on privacy methods in federated learning",
    "suggested_sources": ["arXiv"],
    "domain": "Machine Learning"
}}

Query: "Cancer treatment using machine learning"
Response: {{
    "primary_keywords": ["cancer treatment", "machine learning", "oncology AI"],
    "secondary_keywords": ["tumor detection", "medical AI", "clinical ML"],
    "domain_specific_terms": ["radiotherapy", "chemotherapy", "biomarker", "medical imaging"],
    "temporal_indicators": ["recent", "current"],
    "methodology_keywords": ["deep learning", "neural networks", "classification", "prediction"],
    "comparison_terms": [],
    "optimized_query": "machine learning cancer treatment oncology artificial intelligence",
    "search_strategy": "Search both arXiv for ML methods and PubMed for clinical applications",
    "suggested_sources": ["arXiv", "PubMed"],
    "domain": "Medical AI"
}}

Now extract keywords for the given query. Focus on:
1. Technical accuracy and completeness
2. Academic database optimization
3. Comprehensive coverage of the topic
4. Appropriate source selection

Respond with ONLY the JSON structure."""
        
        try:
            response = self.llm_client.generate_response(
                keyword_extraction_prompt,
                temperature=0.2,  # Lower temperature for more consistent extraction
                max_tokens=800
            )
            
            # Parse the JSON response
            import json
            keywords_data = json.loads(response.strip())
            
            # Validate and ensure all required fields exist
            required_fields = [
                "primary_keywords", "secondary_keywords", "domain_specific_terms",
                "temporal_indicators", "methodology_keywords", "comparison_terms",
                "optimized_query", "search_strategy", "suggested_sources", "domain"
            ]
            
            for field in required_fields:
                if field not in keywords_data:
                    keywords_data[field] = []
            
            logger.info(f"✅ Extracted keywords: {keywords_data['primary_keywords'][:3]}...")
            logger.info(f"📊 Optimized query: {keywords_data['optimized_query']}")
            logger.info(f"🎯 Suggested sources: {keywords_data['suggested_sources']}")
            
            return keywords_data
            
        except Exception as e:
            logger.error(f"❌ Error extracting keywords: {e}")
            
            # Fallback: Create basic keyword structure
            fallback_keywords = {
                "primary_keywords": query.split()[:3],
                "secondary_keywords": [],
                "domain_specific_terms": [],
                "temporal_indicators": ["recent", "current"] if any(word in query.lower() for word in ["recent", "latest", "new", "current"]) else [],
                "methodology_keywords": ["method", "approach"] if any(word in query.lower() for word in ["method", "approach", "technique"]) else [],
                "comparison_terms": ["comparison"] if any(word in query.lower() for word in ["compare", "vs", "versus"]) else [],
                "optimized_query": query,
                "search_strategy": "Direct search with original query terms",
                "suggested_sources": ["arXiv", "PubMed"],
                "domain": "General Research"
            }
            
            return fallback_keywords
    
    def generate_direct_answer(self, query: str) -> str:
        """
        Generate a direct answer for general queries with intelligent question-type adaptation
        
        Args:
            query: User's general question
            
        Returns:
            Direct answer from LLM adapted to question type
        """
        logger.info(f"💬 Generating adaptive direct answer for: {query[:50]}...")
        
        # Analyze question type for adaptive response
        question_type = self._analyze_question_type(query)
        
        # Select appropriate prompt based on question type
        if question_type == "comparison":
            prompt = self._create_comparison_prompt(query)
        elif question_type == "definition":
            prompt = self._create_definition_prompt(query)
        elif question_type == "technical":
            prompt = self._create_technical_prompt(query)
        elif question_type == "factual":
            prompt = self._create_factual_prompt(query)
        elif question_type == "conversational":
            prompt = self._create_conversational_prompt(query)
        else:
            prompt = self._create_general_prompt(query)
        
        try:
            answer = self.llm_client.generate_response(
                prompt,
                temperature=0.3,
                max_tokens=600
            )
            
            logger.info(f"✅ Adaptive answer generated (type: {question_type})")
            return answer.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating direct answer: {e}")
            return "I apologize, but I encountered an error while generating an answer. Please try rephrasing your question or consider running a research analysis."
    
    def _analyze_question_type(self, query: str) -> str:
        """
        Analyze the type of question to determine the best response approach
        
        Args:
            query: User's question
            
        Returns:
            Question type category
        """
        query_lower = query.lower().strip()
        
        # Comparison questions
        comparison_indicators = ['difference between', 'compare', 'vs', 'versus', 'better than', 'differ from']
        if any(indicator in query_lower for indicator in comparison_indicators):
            return "comparison"
        
        # Definition questions
        definition_indicators = ['what is', 'what are', 'define', 'meaning of', 'definition of']
        if any(indicator in query_lower for indicator in definition_indicators):
            # Check if it's technical vs general
            tech_terms = ['api', 'framework', 'algorithm', 'model', 'neural', 'machine learning', 'ai', 'programming', 'database', 'software']
            if any(term in query_lower for term in tech_terms):
                return "technical"
            else:
                return "definition"
        
        # Factual questions (dates, events, people)
        factual_indicators = ['when', 'where', 'who', 'which year', 'what year', 'what date']
        if any(indicator in query_lower for indicator in factual_indicators):
            return "factual"
        
        # Conversational/greeting
        conversational_patterns = ['hi', 'hello', 'how are you', 'thanks', 'thank you']
        if any(pattern in query_lower for pattern in conversational_patterns):
            return "conversational"
        
        # Technical how-to questions
        if query_lower.startswith('how') and any(term in query_lower for term in ['work', 'implement', 'use', 'build', 'create']):
            return "technical"
        
        return "general"
    
    def _create_comparison_prompt(self, query: str) -> str:
        """Create prompt for comparison questions"""
        return f"""You are an expert analyst. Compare the two concepts mentioned in this question clearly and concisely.

Question: {query}

Provide a comparison that includes:
- Brief overview of each concept
- Key differences between them
- Use cases or contexts where each excels
- Practical considerations for choosing between them

Keep your response conversational and well-structured, but avoid rigid numbered lists unless the comparison naturally fits that format."""

    def _create_definition_prompt(self, query: str) -> str:
        """Create prompt for definition questions about general topics"""
        return f"""You are a knowledgeable educator. Provide a clear, accessible explanation of the concept being asked about.

Question: {query}

Provide an explanation that includes:
- A clear, simple definition
- Why it's important or relevant
- Key aspects or characteristics
- Real-world context or examples if helpful

Keep your answer informative but approachable, as if explaining to someone genuinely curious about the topic."""

    def _create_technical_prompt(self, query: str) -> str:
        """Create prompt for technical questions"""
        return f"""You are a technical expert. Provide a comprehensive but accessible explanation of the technical concept being asked about.

Question: {query}

Your response should include:
- Clear explanation of the concept
- Key technical aspects and components
- How it works or is implemented
- Common use cases and applications
- Important considerations or limitations

Balance technical accuracy with clarity. If this is an emerging or rapidly evolving technology, mention that recent developments may require consulting current research."""

    def _create_factual_prompt(self, query: str) -> str:
        """Create prompt for factual questions"""
        return f"""You are a knowledgeable reference source. Provide accurate, factual information about what's being asked.

Question: {query}

Provide a straightforward, informative answer that includes:
- Direct answer to the question
- Relevant context or background
- Key facts or details
- Any important related information

Keep your response factual and well-organized without unnecessary academic formatting."""

    def _create_conversational_prompt(self, query: str) -> str:
        """Create prompt for conversational/greeting questions"""
        return f"""You are a friendly, helpful AI assistant. Respond naturally to this conversational query.

Message: {query}

Respond in a warm, conversational manner while being helpful and informative if the user is seeking any information."""

    def _create_general_prompt(self, query: str) -> str:
        """Create prompt for general questions that don't fit other categories"""
        return f"""You are a knowledgeable assistant. Provide a helpful, informative response to this question.

Question: {query}

Provide a clear, useful answer that addresses what the user is asking about. Structure your response naturally based on the type of information being requested."""
    
    def is_research_query(self, query: str) -> bool:
        """
        Simple boolean check if query requires research
        
        Args:
            query: User's question
            
        Returns:
            True if research needed, False for direct answer
        """
        return self.classify_query(query) == "research"
    
    def get_query_intent(self, query: str) -> dict:
        """
        Extract more detailed intent information from the query
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with intent details
        """
        
        intent_prompt = f"""
        Analyze this research query and extract key information:

        Query: "{query}"

        Provide a JSON response with:
        {{
            "type": "general" or "research",
            "domain": "field of study (e.g., NLP, computer vision, biology)",
            "time_focus": "recent", "historical", or "general",
            "comparison_needed": true/false,
            "specific_concepts": ["list", "of", "key", "terms"]
        }}

        Respond with only the JSON.
        """
        
        try:
            response = self.llm_client.generate_response(
                intent_prompt,
                temperature=0.1,
                max_tokens=200
            )
            
            import json
            intent_data = json.loads(response.strip())
            return intent_data
            
        except Exception as e:
            # Return default intent structure
            return {
                "type": "research",
                "domain": "general",
                "time_focus": "recent",
                "comparison_needed": False,
                "specific_concepts": query.split()[:3]
            } 