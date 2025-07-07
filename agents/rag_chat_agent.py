"""
RAG Chat Agent for Co-Researcher.AI
===================================

This agent handles follow-up questions about research papers using 
Retrieval-Augmented Generation (RAG). It searches the vector store for 
relevant paper content and generates grounded, citation-backed answers.

Features:
- Context-aware question answering
- Automatic citation generation
- Conversation memory for multi-turn chat
- Source paper tracking and linking

Author: Co-Researcher.AI Team
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RAGChatAgent:
    """
    Retrieval-Augmented Generation chat agent for research Q&A
    
    This agent answers user questions by retrieving relevant content from
    analyzed papers and generating responses grounded in that content.
    """
    
    def __init__(self, llm_client, vector_store):
        """
        Initialize the RAG chat agent
        
        Args:
            llm_client: Instance of LLMClient for generating responses
            vector_store: Instance of VectorStore for retrieving relevant content
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.conversation_history = []  # Store conversation context
        
        logger.info("💬 RAG Chat Agent initialized")
    
    def answer_question(
        self, 
        question: str, 
        max_context_length: int = 4000,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using retrieved paper content
        
        Args:
            question (str): User's question about the research
            max_context_length (int): Maximum length of context to use
            include_sources (bool): Whether to include source citations
            
        Returns:
            Dict: Response with answer, sources, and metadata
        """
        logger.info(f"❓ Answering question: {question[:100]}...")
        
        try:
            # Retrieve relevant context from the vector store
            context, source_papers = self.vector_store.get_context_for_query(
                question, max_context_length
            )
            
            # Generate answer using retrieved context
            answer = self._generate_answer_with_context(question, context)
            
            # Format response with sources
            response = {
                'answer': answer,
                'question': question,
                'sources': source_papers if include_sources else [],
                'context_used': len(context),
                'timestamp': datetime.now().isoformat(),
                'has_sources': len(source_papers) > 0
            }
            
            # Add to conversation history
            self.conversation_history.append({
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"✅ Answer generated (using {len(source_papers)} sources)")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error answering question: {e}")
            return self._create_error_response(question, str(e))
    
    def _generate_answer_with_context(self, question: str, context: str) -> str:
        """
        Generate an answer using the retrieved context
        
        Args:
            question (str): User's question
            context (str): Retrieved relevant content from papers
            
        Returns:
            str: Generated answer
        """
        # Create a prompt that emphasizes using the provided context
        prompt = f"""You are a research assistant helping users understand academic papers. Answer the user's question based ONLY on the provided research context. Provide comprehensive, detailed answers.

Research Context from Papers:
{context}

User Question: {question}

Instructions:
1. Answer the question using ONLY information from the provided research context
2. Provide detailed, comprehensive answers with specific information from the papers
3. Include author names when mentioning findings (e.g., "According to Smith et al. in '[Paper Title]'...")
4. If asking about authors specifically, extract and list all author names mentioned in the context
5. If multiple papers discuss the topic, provide detailed comparisons
6. Use bullet points or numbered lists for clarity when appropriate
7. Quote relevant passages when they directly answer the question
8. If the context doesn't contain enough information, explain what information is missing
9. Be thorough and informative - users want comprehensive answers

Answer:"""
        
        try:
            answer = self.llm_client.generate_response(
                prompt,
                temperature=0.2,  # Lower temperature for more factual responses
                max_tokens=1200  # Increased for more comprehensive answers
            )
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer. Please try rephrasing your question."
    
    def _create_error_response(self, question: str, error_msg: str) -> Dict[str, Any]:
        """
        Create an error response when answer generation fails
        
        Args:
            question (str): Original question
            error_msg (str): Error message
            
        Returns:
            Dict: Error response structure
        """
        return {
            'answer': "I apologize, but I encountered an error while processing your question. This could be due to technical issues or if no relevant research content is available.",
            'question': question,
            'sources': [],
            'context_used': 0,
            'timestamp': datetime.now().isoformat(),
            'has_sources': False,
            'error': error_msg
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current conversation
        
        Returns:
            Dict: Conversation statistics and recent questions
        """
        recent_questions = [
            entry['question'] for entry in self.conversation_history[-5:]
        ]
        
        return {
            'total_questions': len(self.conversation_history),
            'recent_questions': recent_questions,
            'session_start': self.conversation_history[0]['timestamp'] if self.conversation_history else None,
            'last_activity': self.conversation_history[-1]['timestamp'] if self.conversation_history else None
        }
    
    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("🗑️ Conversation history cleared")
    
    def suggest_questions(self, papers: List[Dict[str, Any]]) -> List[str]:
        """
        Suggest relevant questions based on the analyzed papers
        
        Args:
            papers (List[Dict]): List of analyzed papers
            
        Returns:
            List[str]: Suggested questions
        """
        logger.info("💡 Generating suggested questions based on papers...")
        
        try:
            # Extract key topics from papers
            topics = set()
            methods = set()
            
            for paper in papers[:5]:  # Use top 5 papers
                title = paper.get('title', '')
                
                # Extract key terms from titles
                title_words = title.lower().split()
                
                # Common research topics
                research_topics = ['learning', 'network', 'model', 'algorithm', 'system', 
                                 'method', 'approach', 'framework', 'analysis', 'optimization']
                
                for word in title_words:
                    if len(word) > 4 and word in research_topics:
                        topics.add(word)
            
            # Generate suggested questions
            suggestions = []
            
            if topics:
                topic_list = list(topics)[:3]
                suggestions.extend([
                    f"What are the main approaches to {topic_list[0] if topic_list else 'this research area'}?",
                    f"How do the papers compare different {topic_list[1] if len(topic_list) > 1 else 'methods'}?",
                    f"What are the limitations mentioned in the {topic_list[2] if len(topic_list) > 2 else 'research'}?"
                ])
            
            # Add general questions
            suggestions.extend([
                "What are the key findings across all papers?",
                "Which paper has the most novel contribution?",
                "What future research directions are suggested?",
                "What methodologies are most commonly used?",
                "How do the results compare between different studies?"
            ])
            
            return suggestions[:6]  # Return top 6 suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return [
                "What are the main research findings?",
                "How do the papers relate to each other?",
                "What are the key contributions?",
                "What limitations are mentioned?",
                "What future work is suggested?"
            ]
    
    def search_papers(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for papers relevant to a query using the vector store
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: Relevant paper chunks with metadata
        """
        try:
            search_results = self.vector_store.search(query, top_k * 2)  # Get extra for deduplication
            
            # Group results by paper and take top unique papers
            papers_seen = set()
            unique_results = []
            
            for chunk_text, metadata, similarity_score in search_results:
                paper_title = metadata['paper_title']
                
                if paper_title not in papers_seen and len(unique_results) < top_k:
                    papers_seen.add(paper_title)
                    unique_results.append({
                        'title': metadata['paper_title'],
                        'authors': metadata['paper_authors'],
                        'url': metadata['paper_url'],
                        'published_date': metadata['paper_published_date'],
                        'relevance_score': similarity_score,
                        'excerpt': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                    })
            
            return unique_results
            
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
    
    def explain_findings(self, paper_title: str) -> Dict[str, Any]:
        """
        Provide a detailed explanation of a specific paper's findings
        
        Args:
            paper_title (str): Title of the paper to explain
            
        Returns:
            Dict: Detailed explanation of the paper's findings
        """
        try:
            # Search for content related to this specific paper
            paper_query = f"findings results conclusions {paper_title}"
            context, sources = self.vector_store.get_context_for_query(paper_query, 1500)
            
            if not context:
                return {
                    'explanation': f"I couldn't find detailed information about '{paper_title}' in the analyzed papers.",
                    'paper_title': paper_title,
                    'found_content': False
                }
            
            # Generate detailed explanation
            prompt = f"""Explain the key findings and contributions of this research paper in detail:

Paper Title: {paper_title}

Research Content:
{context}

Provide a comprehensive explanation covering:
1. Main research objectives
2. Key findings and results
3. Methodology used
4. Significance and implications
5. Any limitations mentioned

Be specific and cite the exact findings from the paper."""
            
            explanation = self.llm_client.generate_response(
                prompt,
                temperature=0.2,
                max_tokens=800
            )
            
            return {
                'explanation': explanation.strip(),
                'paper_title': paper_title,
                'found_content': True,
                'sources': sources
            }
            
        except Exception as e:
            logger.error(f"Error explaining findings for '{paper_title}': {e}")
            return {
                'explanation': f"I encountered an error while analyzing '{paper_title}'. Please try again.",
                'paper_title': paper_title,
                'found_content': False,
                'error': str(e)
            } 