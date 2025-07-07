"""
Validation Agent
Responsible for assessing citation count, recency, and source reliability of research papers.
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import sys
import os
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class ValidationAgent:
    """
    Agent responsible for validating and scoring research papers based on various criteria
    """
    
    def __init__(self):
        """Initialize the validation agent"""
        logger.info("Validation Agent initialized")
        
        # Scoring weights for different criteria
        self.weights = {
            'recency': 0.3,      # How recent the paper is
            'venue_quality': 0.2, # Quality of publication venue
            'citation_count': 0.25, # Number of citations (if available)
            'author_reputation': 0.15, # Reputation of authors
            'content_quality': 0.1   # Quality indicators from content
        }
    
    def validate_paper(self, paper: 'ResearchPaper') -> float:
        """
        Validate a research paper and return a quality score (0-1)
        
        Args:
            paper: ResearchPaper object to validate
            
        Returns:
            Validation score between 0 and 1
        """
        logger.info(f"Validating paper: {paper.title[:50]}...")
        
        try:
            scores = {}
            
            # Score different aspects
            scores['recency'] = self._score_recency(paper)
            scores['venue_quality'] = self._score_venue_quality(paper)
            scores['citation_count'] = self._score_citations(paper)
            scores['author_reputation'] = self._score_author_reputation(paper)
            scores['content_quality'] = self._score_content_quality(paper)
            
            # Calculate weighted average
            total_score = sum(
                self.weights[criterion] * score 
                for criterion, score in scores.items()
            )
            
            logger.info(f"Validation complete. Score: {total_score:.3f}")
            logger.debug(f"Score breakdown: {scores}")
            
            return round(total_score, 3)
            
        except Exception as e:
            logger.error(f"Error validating paper: {str(e)}")
            return 0.5  # Default neutral score
    
    def _score_recency(self, paper: 'ResearchPaper') -> float:
        """
        Score paper based on how recent it is
        
        Args:
            paper: ResearchPaper object
            
        Returns:
            Score between 0 and 1 (1 = very recent)
        """
        try:
            # Parse publication date
            pub_date = datetime.fromisoformat(paper.published_date.replace('Z', '+00:00'))
            current_date = datetime.now(pub_date.tzinfo)
            
            # Calculate age in days
            age_days = (current_date - pub_date).days
            
            # Score based on age (newer = higher score)
            if age_days <= 30:  # Very recent (last month)
                return 1.0
            elif age_days <= 180:  # Recent (last 6 months)
                return 0.9
            elif age_days <= 365:  # Somewhat recent (last year)
                return 0.7
            elif age_days <= 730:  # Within 2 years
                return 0.5
            elif age_days <= 1095:  # Within 3 years
                return 0.3
            else:  # Older than 3 years
                return 0.1
                
        except Exception as e:
            logger.error(f"Error scoring recency: {str(e)}")
            return 0.5  # Default score
    
    def _score_venue_quality(self, paper: 'ResearchPaper') -> float:
        """
        Score based on publication venue quality
        
        For arXiv papers, we use heuristics based on the paper content
        """
        try:
            # Since we're dealing with arXiv preprints, score based on indicators
            title = paper.title.lower()
            abstract = paper.abstract.lower()
            
            quality_indicators = {
                'high': [
                    'neurips', 'icml', 'iclr', 'aaai', 'ijcai', 'acl', 'emnlp',
                    'cvpr', 'iccv', 'eccv', 'sigkdd', 'www', 'chi', 'uist'
                ],
                'medium': [
                    'workshop', 'conference', 'symposium', 'journal'
                ],
                'research_quality': [
                    'novel', 'state-of-the-art', 'breakthrough', 'significant',
                    'comprehensive', 'extensive', 'rigorous', 'systematic'
                ]
            }
            
            # Check for high-quality venue mentions
            text = f"{title} {abstract}"
            
            for venue in quality_indicators['high']:
                if venue in text:
                    return 0.9
            
            for indicator in quality_indicators['research_quality']:
                if indicator in text:
                    return 0.8
            
            for venue in quality_indicators['medium']:
                if venue in text:
                    return 0.7
            
            # Default score for arXiv papers
            return 0.6
            
        except Exception as e:
            logger.error(f"Error scoring venue quality: {str(e)}")
            return 0.6
    
    def _score_citations(self, paper: 'ResearchPaper') -> float:
        """
        Score based on citation count (if available)
        
        Since we're using arXiv, we'll estimate based on other factors
        """
        try:
            # For arXiv papers, we don't have direct citation data
            # Use age and content quality as proxies
            
            # Parse publication date to estimate potential citations
            pub_date = datetime.fromisoformat(paper.published_date.replace('Z', '+00:00'))
            current_date = datetime.now(pub_date.tzinfo)
            age_months = (current_date - pub_date).days / 30
            
            # Estimate citation potential based on content indicators
            abstract = paper.abstract.lower()
            title = paper.title.lower()
            
            citation_indicators = [
                'novel', 'first', 'new', 'breakthrough', 'significant',
                'outperform', 'state-of-the-art', 'benchmark', 'evaluation',
                'comprehensive', 'extensive', 'systematic'
            ]
            
            indicator_count = sum(1 for indicator in citation_indicators 
                                if indicator in f"{title} {abstract}")
            
            # Base score on indicators and age
            base_score = min(indicator_count * 0.15, 0.8)
            
            # Adjust for age (older papers have more time to accumulate citations)
            if age_months > 12:
                age_bonus = min(age_months / 24, 0.2)
                base_score += age_bonus
            
            return min(base_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error scoring citations: {str(e)}")
            return 0.5
    
    def _score_author_reputation(self, paper: 'ResearchPaper') -> float:
        """
        Score based on author reputation (simplified heuristics)
        """
        try:
            authors = paper.authors
            
            # Heuristics for author quality
            quality_score = 0.5  # Base score
            
            # More authors might indicate collaboration (positive)
            if len(authors) >= 3:
                quality_score += 0.1
            elif len(authors) >= 5:
                quality_score += 0.2
            
            # Check for institutional indicators in author names/affiliations
            # (This is limited since we only have names from arXiv)
            
            # Check if any authors have institutional email patterns
            institutional_indicators = [
                '.edu', '.ac.', 'university', 'institute', 'lab',
                'google', 'microsoft', 'facebook', 'openai', 'deepmind'
            ]
            
            author_text = ' '.join(authors).lower()
            for indicator in institutional_indicators:
                if indicator in author_text:
                    quality_score += 0.1
                    break
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error scoring author reputation: {str(e)}")
            return 0.5
    
    def _score_content_quality(self, paper: 'ResearchPaper') -> float:
        """
        Score based on content quality indicators
        """
        try:
            title = paper.title.lower()
            abstract = paper.abstract.lower()
            
            quality_indicators = {
                'methodology': [
                    'evaluation', 'benchmark', 'experiment', 'analysis',
                    'systematic', 'comprehensive', 'extensive', 'rigorous'
                ],
                'innovation': [
                    'novel', 'new', 'first', 'breakthrough', 'innovative',
                    'original', 'unique', 'pioneering'
                ],
                'impact': [
                    'significant', 'important', 'substantial', 'major',
                    'considerable', 'notable', 'remarkable'
                ],
                'technical': [
                    'algorithm', 'model', 'framework', 'architecture',
                    'method', 'approach', 'technique', 'system'
                ]
            }
            
            score = 0.3  # Base score
            text = f"{title} {abstract}"
            
            # Score each category
            for category, indicators in quality_indicators.items():
                category_score = sum(1 for indicator in indicators if indicator in text)
                score += min(category_score * 0.1, 0.2)  # Max 0.2 per category
            
            # Bonus for length (longer abstracts often indicate more detail)
            if len(abstract) > 1000:
                score += 0.1
            elif len(abstract) > 500:
                score += 0.05
            
            # Check for quantitative results
            if re.search(r'\d+(?:\.\d+)?%', abstract) or re.search(r'\d+(?:\.\d+)?\s*(?:accuracy|precision|recall|f1)', abstract):
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error scoring content quality: {str(e)}")
            return 0.5
    
    def get_validation_details(self, paper: 'ResearchPaper') -> Dict[str, Any]:
        """
        Get detailed validation breakdown for a paper
        
        Args:
            paper: ResearchPaper object
            
        Returns:
            Dictionary with detailed validation metrics
        """
        try:
            details = {
                'total_score': self.validate_paper(paper),
                'recency_score': self._score_recency(paper),
                'venue_quality_score': self._score_venue_quality(paper),
                'citation_score': self._score_citations(paper),
                'author_reputation_score': self._score_author_reputation(paper),
                'content_quality_score': self._score_content_quality(paper),
                'weights_used': self.weights,
                'paper_age_days': self._get_paper_age_days(paper),
                'author_count': len(paper.authors)
            }
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting validation details: {str(e)}")
            return {'error': str(e)}
    
    def _get_paper_age_days(self, paper: 'ResearchPaper') -> int:
        """Get paper age in days"""
        try:
            pub_date = datetime.fromisoformat(paper.published_date.replace('Z', '+00:00'))
            current_date = datetime.now(pub_date.tzinfo)
            return (current_date - pub_date).days
        except:
            return 0 