"""
Paper Reranker Module for Co-Researcher.AI
==========================================

This module implements a hybrid ranking system that combines:
1. Fast embedding-based semantic similarity (first pass)
2. Deep LLM-based relevance scoring (second pass)

This two-stage approach ensures we get the most relevant papers while being cost-effective.

Author: Co-Researcher.AI Team
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class PaperReranker:
    """
    Hybrid paper ranking system using embeddings + LLM reranking
    
    Pipeline:
    1. Use sentence embeddings to quickly filter papers by semantic similarity
    2. Use LLM to deeply analyze the top candidates and provide final ranking
    3. Return the top N most relevant papers with explanations
    """
    
    def __init__(self, llm_client, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the reranker with embedding model and LLM client
        
        Args:
            llm_client: Instance of LLMClient for deep analysis
            embedding_model (str): HuggingFace model for embeddings
        """
        self.llm_client = llm_client
        self.embedding_model_name = embedding_model
        
        # Load the sentence transformer model for semantic similarity
        logger.info(f"📊 Loading embedding model: {embedding_model}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            np.ndarray: Matrix of embeddings
        """
        try:
            logger.info(f"🔢 Computing embeddings for {len(texts)} texts...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            logger.info("✅ Embeddings computed successfully")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Error computing embeddings: {e}")
            raise
    
    def _cosine_similarity(self, query_embedding: np.ndarray, paper_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and paper embeddings
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            paper_embeddings (np.ndarray): Matrix of paper embeddings
            
        Returns:
            np.ndarray: Similarity scores for each paper
        """
        # Normalize vectors for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        paper_norms = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarities = np.dot(paper_norms, query_norm)
        return similarities
    
    def _embedding_based_filtering(
        self, 
        query: str, 
        papers: List[Dict[str, Any]], 
        top_k: int = 15
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        First pass: Use embeddings to quickly filter papers by semantic similarity
        
        This fast filtering step reduces the number of papers we need to send to the
        expensive LLM for detailed analysis.
        
        Args:
            query (str): User's research query
            papers (List[Dict]): List of paper dictionaries
            top_k (int): Number of top papers to return
            
        Returns:
            List[Tuple[Dict, float]]: Papers with their similarity scores
        """
        logger.info(f"🎯 Stage 1: Embedding-based filtering (top {top_k})")
        
        # Prepare texts for embedding: combine title and abstract
        paper_texts = []
        for paper in papers:
            # Create a comprehensive text representation of each paper
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            text = f"{title}\n\n{abstract}"
            paper_texts.append(text)
        
        # Compute embeddings for query and papers
        query_embedding = self._compute_embeddings([query])[0]
        paper_embeddings = self._compute_embeddings(paper_texts)
        
        # Calculate similarity scores
        similarities = self._cosine_similarity(query_embedding, paper_embeddings)
        
        # Sort papers by similarity score
        paper_similarity_pairs = list(zip(papers, similarities))
        sorted_pairs = sorted(paper_similarity_pairs, key=lambda x: x[1], reverse=True)
        
        # Return top K papers with their scores
        top_papers = sorted_pairs[:top_k]
        
        logger.info(f"✅ Filtered to top {len(top_papers)} papers by embedding similarity")
        for i, (paper, score) in enumerate(top_papers[:3]):
            logger.info(f"   {i+1}. {paper['title'][:50]}... (score: {score:.3f})")
        
        return top_papers
    
    def _llm_based_reranking(
        self, 
        query: str, 
        papers_with_scores: List[Tuple[Dict[str, Any], float]], 
        final_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Second pass: Use LLM to deeply analyze and rerank the top papers
        
        This provides more sophisticated relevance assessment than embeddings alone.
        
        Args:
            query (str): User's research query
            papers_with_scores (List[Tuple]): Papers with embedding similarity scores
            final_k (int): Final number of papers to return
            
        Returns:
            List[Dict]: Reranked papers with relevance explanations
        """
        logger.info(f"🧠 Stage 2: LLM-based reranking (top {final_k})")
        
        # Prepare paper information for LLM analysis
        paper_summaries = []
        for i, (paper, embedding_score) in enumerate(papers_with_scores):
            paper_info = f"""Paper {i+1}:
Title: {paper['title']}
Authors: {', '.join(paper.get('authors', [])[:3])}
Published: {paper.get('published_date', 'Unknown')[:10]}
Abstract: {paper.get('abstract', '')[:500]}...
Embedding Score: {embedding_score:.3f}
"""
            paper_summaries.append(paper_info)
        
        # Create LLM prompt for detailed relevance analysis with quality scoring
        reranking_prompt = f"""You are an expert research analyst. Given this research query and list of papers, rank them by relevance AND provide quality scores on a 1-10 scale.

Research Query: "{query}"

Papers to analyze:
{chr(10).join(paper_summaries)}

Instructions:
1. Analyze each paper's relevance to the query
2. Consider: topic alignment, research quality, recency, methodology, impact
3. Rank the top {final_k} most relevant papers
4. Assign a relevance score (1-10 scale) where:
   - 9-10: Excellent match, directly addresses query
   - 7-8: Good relevance, strong connection to topic
   - 5-6: Moderate relevance, some useful insights
   - 3-4: Limited relevance, tangential connection
   - 1-2: Poor match, barely related

Format your response EXACTLY as:
1. Paper [Number]: [Score]/10 - [Brief explanation]
2. Paper [Number]: [Score]/10 - [Brief explanation]
...

Example:
1. Paper 3: 8.5/10 - Excellent methodology for machine learning applications
2. Paper 1: 6.5/10 - Good background research, somewhat related approach

Focus on papers that directly address the query and have strong research contributions."""
        
        try:
            # Get LLM analysis
            llm_response = self.llm_client.generate_response(
                reranking_prompt,
                temperature=0.3,
                max_tokens=800
            )
            
            # Parse LLM response to extract rankings
            ranked_papers = self._parse_llm_rankings(llm_response, papers_with_scores, final_k)
            
            logger.info(f"✅ LLM reranking complete, selected {len(ranked_papers)} papers")
            
            return ranked_papers
            
        except Exception as e:
            logger.error(f"❌ LLM reranking failed: {e}")
            # Fallback: return top papers by embedding score
            logger.info("📋 Falling back to embedding-based ranking")
            return [paper for paper, score in papers_with_scores[:final_k]]
    
    def _parse_llm_rankings(
        self, 
        llm_response: str, 
        papers_with_scores: List[Tuple[Dict[str, Any], float]], 
        final_k: int
    ) -> List[Dict[str, Any]]:
        """
        Parse the LLM's ranking response and extract quality scores
        
        Args:
            llm_response (str): LLM's ranking response with quality scores
            papers_with_scores (List[Tuple]): Original papers with scores
            final_k (int): Number of papers to return
            
        Returns:
            List[Dict]: Reranked papers with LLM-assigned quality scores
        """
        ranked_papers = []
        
        try:
            # Extract paper numbers and quality scores from LLM response
            lines = llm_response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or len(ranked_papers) >= final_k:
                    continue
                
                # Look for patterns like "1. Paper 3: 8.5/10 - explanation"
                import re
                match = re.match(r'^\d+\.\s*(?:Paper\s*)?(\d+):\s*([\d.]+)/10\s*-\s*(.+)$', line)
                
                if match:
                    paper_number = int(match.group(1)) - 1  # Convert to 0-based index
                    quality_score = float(match.group(2))   # Extract LLM quality score
                    explanation = match.group(3).strip()    # Extract explanation
                    
                    # Validate paper number and quality score
                    if 0 <= paper_number < len(papers_with_scores) and 0 <= quality_score <= 10:
                        paper, embedding_score = papers_with_scores[paper_number]
                        
                        # Add paper with LLM-assigned quality score
                        enhanced_paper = paper.copy()
                        enhanced_paper['relevance_score'] = quality_score  # Use LLM quality score!
                        enhanced_paper['embedding_score'] = embedding_score
                        enhanced_paper['relevance_explanation'] = explanation
                        
                        ranked_papers.append(enhanced_paper)
                        logger.info(f"   📝 Ranked paper {len(ranked_papers)}: {paper['title'][:50]}... (quality: {quality_score}/10)")
                else:
                    # Fallback: try to parse without score format
                    fallback_match = re.match(r'^\d+\.\s*(?:Paper\s*)?(\d+)', line)
                    if fallback_match:
                        paper_number = int(fallback_match.group(1)) - 1
                        
                        if 0 <= paper_number < len(papers_with_scores):
                            paper, embedding_score = papers_with_scores[paper_number]
                            
                            # Extract explanation
                            explanation_match = re.search(r'-\s*(.+)$', line)
                            explanation = explanation_match.group(1) if explanation_match else "Relevant to query"
                            
                            # Assign position-based score as fallback (decreasing quality)
                            fallback_score = max(1.0, 10.0 - len(ranked_papers) * 1.5)
                            
                            enhanced_paper = paper.copy()
                            enhanced_paper['relevance_score'] = fallback_score
                            enhanced_paper['embedding_score'] = embedding_score
                            enhanced_paper['relevance_explanation'] = explanation.strip()
                            
                            ranked_papers.append(enhanced_paper)
                            logger.info(f"   📝 Ranked paper {len(ranked_papers)} (fallback): {paper['title'][:50]}... (score: {fallback_score}/10)")
            
            # If we couldn't parse enough papers, fill with embedding-based fallbacks
            while len(ranked_papers) < final_k and len(ranked_papers) < len(papers_with_scores):
                next_index = len(ranked_papers)
                if next_index < len(papers_with_scores):
                    paper, embedding_score = papers_with_scores[next_index]
                    
                    # Assign decreasing quality scores for remaining papers
                    fallback_score = max(1.0, 8.0 - next_index * 1.0)
                    
                    enhanced_paper = paper.copy()
                    enhanced_paper['relevance_score'] = fallback_score
                    enhanced_paper['embedding_score'] = embedding_score
                    enhanced_paper['relevance_explanation'] = "Selected by embedding similarity"
                    ranked_papers.append(enhanced_paper)
                    logger.info(f"   📝 Added paper {len(ranked_papers)} (embedding): {paper['title'][:50]}... (score: {fallback_score}/10)")
                else:
                    break
            
        except Exception as e:
            logger.error(f"Error parsing LLM rankings: {e}")
            # Fallback: return papers with reasonable quality scores
            ranked_papers = []
            for i, (paper, embedding_score) in enumerate(papers_with_scores[:final_k]):
                # Assign reasonable quality scores: 8.0, 7.0, 6.0, 5.0, 4.0
                fallback_score = max(1.0, 8.0 - i * 1.0)
                
                enhanced_paper = paper.copy()
                enhanced_paper['relevance_score'] = fallback_score
                enhanced_paper['embedding_score'] = embedding_score
                enhanced_paper['relevance_explanation'] = "Ranked by embedding similarity (parsing failed)"
                ranked_papers.append(enhanced_paper)
                logger.info(f"   📝 Fallback paper {i+1}: {paper['title'][:50]}... (score: {fallback_score}/10)")
        
        # Ensure all papers have valid relevance scores (1-10 range)
        for paper in ranked_papers:
            if 'relevance_score' not in paper or paper['relevance_score'] <= 0:
                paper['relevance_score'] = 5.0  # Default moderate score
                
        return ranked_papers
    
    def rank_papers(
        self, 
        query: str, 
        papers: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Main entry point: Rank papers using the hybrid approach
        
        This method orchestrates the two-stage ranking process:
        1. Fast embedding-based filtering
        2. Deep LLM-based reranking
        
        Args:
            query (str): User's research query
            papers (List[Dict]): List of papers to rank
            top_k (int): Number of top papers to return
            
        Returns:
            List[Dict]: Top K papers ranked by relevance
        """
        logger.info(f"🎯 Starting hybrid paper ranking for {len(papers)} papers")
        
        if not papers:
            logger.warning("No papers provided for ranking")
            return []
        
        if len(papers) <= top_k:
            logger.info(f"Only {len(papers)} papers available, returning all")
            # Ensure all papers have quality scores (1-10 scale)
            enhanced_papers = []
            for i, paper in enumerate(papers):
                enhanced_paper = paper.copy()
                # Assign reasonable quality scores: start at 8.0 and decrease
                enhanced_paper['relevance_score'] = max(1.0, 8.0 - i * 0.5)
                if 'relevance_explanation' not in enhanced_paper:
                    enhanced_paper['relevance_explanation'] = "Limited papers available"
                if 'embedding_score' not in enhanced_paper:
                    enhanced_paper['embedding_score'] = 0.0
                enhanced_papers.append(enhanced_paper)
            return enhanced_papers
        
        try:
            # Stage 1: Embedding-based filtering (3x the final count for LLM analysis)
            embedding_top_k = min(top_k * 3, len(papers))
            filtered_papers = self._embedding_based_filtering(query, papers, embedding_top_k)
            
            # Stage 2: LLM-based reranking of filtered papers
            final_papers = self._llm_based_reranking(query, filtered_papers, top_k)
            
            logger.info(f"🏆 Ranking complete! Selected {len(final_papers)} top papers")
            
            return final_papers
            
        except Exception as e:
            logger.error(f"❌ Error in ranking pipeline: {e}")
            # Fallback: return first N papers with quality scores
            logger.info("📋 Using fallback ranking (first N papers)")
            fallback_papers = []
            for i, paper in enumerate(papers[:top_k]):
                enhanced_paper = paper.copy()
                # Assign moderate quality scores: 7.0, 6.5, 6.0, 5.5, 5.0, etc.
                enhanced_paper['relevance_score'] = max(1.0, 7.0 - i * 0.5)
                enhanced_paper['relevance_explanation'] = "Fallback ranking by order"
                enhanced_paper['embedding_score'] = 0.0  # No embedding score available
                fallback_papers.append(enhanced_paper)
            return fallback_papers 