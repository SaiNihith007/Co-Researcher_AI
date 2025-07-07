"""
Data Collection Agent for Co-Researcher.AI
==========================================

This agent handles fetching research papers from multiple trusted sources:
- arXiv: Computer Science and Physics papers
- PubMed: Biomedical and life sciences papers  
- Semantic Scholar: Cross-disciplinary papers (future feature)

The agent filters papers by date, relevance, and quality before returning them.

Author: Co-Researcher.AI Team
"""

import arxiv
import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import ResearchPaper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class DataCollectionAgent:
    """
    Multi-source research paper collection agent
    
    This agent searches and collects papers from various academic databases,
    applying filters for quality, recency, and relevance.
    """
    
    def __init__(self, years_back: int = 3, router=None, enable_full_papers: bool = True):
        """
        Initialize the data collection agent
        
        Args:
            years_back (int): How many years back to search for papers
            router: Optional QueryRouter instance for intelligent keyword extraction
            enable_full_papers (bool): Whether to process full paper content when available
        """
        self.years_back = years_back
        self.router = router
        self.enable_full_papers = enable_full_papers
        self.arxiv_client = arxiv.Client()
        
        # Initialize PDF processor for full content extraction
        if enable_full_papers:
            from modules.pdf_processor import PDFProcessor
            self.pdf_processor = PDFProcessor()
            logger.info("📄 PDF processor initialized for full paper content")
        else:
            self.pdf_processor = None
        
        logger.info(f"📚 Data Collection Agent initialized (searching {years_back} years back)")
        if router:
            logger.info("🧠 Router integration enabled for intelligent keyword extraction")
    
    def _is_recent_enough(self, published_date: str) -> bool:
        """
        Check if a paper is recent enough based on our criteria
        
        Args:
            published_date (str): ISO date string
            
        Returns:
            bool: True if paper is within our date range
        """
        try:
            pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            cutoff_date = datetime.now(pub_date.tzinfo) - timedelta(days=365 * self.years_back)
            return pub_date >= cutoff_date
        except:
            return True  # Include if we can't parse date
    
    def search_arxiv(self, query: str, max_papers: int = 20) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching the query
        
        Args:
            query (str): Search query
            max_papers (int): Maximum number of papers to fetch
            
        Returns:
            List[Dict]: List of paper dictionaries
        """
        logger.info(f"🔍 Searching arXiv for: '{query}' (max {max_papers} papers)")
        
        papers = []
        
        try:
            # Create arXiv search with query
            search = arxiv.Search(
                query=query,
                max_results=max_papers * 3,  # Increased from * 2 to * 3 to overcome arXiv's recency bias
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            
            # Fetch results and convert to our format
            for result in self.arxiv_client.results(search):
                # Convert arXiv result to standard format
                paper = {
                    'title': result.title.strip(),
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary.strip(),
                    'url': result.entry_id,
                    'published_date': result.published.isoformat(),
                    'arxiv_id': result.get_short_id(),
                    'source': 'arXiv',
                    'categories': [cat for cat in result.categories],
                    'pdf_url': result.pdf_url if hasattr(result, 'pdf_url') else result.entry_id.replace('/abs/', '/pdf/') + '.pdf'
                }
                
                # Apply date filter
                if self._is_recent_enough(paper['published_date']):
                    papers.append(paper)
                    
                    # Stop if we have enough papers
                    if len(papers) >= max_papers:
                        break
            
            logger.info(f"✅ Found {len(papers)} relevant papers from arXiv")
            
        except Exception as e:
            logger.error(f"❌ Error searching arXiv: {e}")
        
        return papers
    
    def search_pubmed(self, query: str, max_papers: int = 20) -> List[Dict[str, Any]]:
        """
        Search PubMed for biomedical papers (using E-utilities API)
        
        Args:
            query (str): Search query
            max_papers (int): Maximum number of papers to fetch
            
        Returns:
            List[Dict]: List of paper dictionaries
        """
        logger.info(f"🔬 Searching PubMed for: '{query}' (max {max_papers} papers)")
        
        papers = []
        
        try:
            # Step 1: Search PubMed for PMIDs
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_papers,
                'sort': 'relevance',
                'retmode': 'json'
            }
            
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            search_data = response.json()
            
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                logger.info("No PubMed results found")
                return papers
            
            # Step 2: Fetch paper details for the PMIDs
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml'
            }
            
            response = requests.get(fetch_url, params=fetch_params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response (basic parsing)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # Extract paper information
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract title
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "No title"
                    
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        last_name = author.find('LastName')
                        first_name = author.find('ForeName')
                        if last_name is not None:
                            name = last_name.text
                            if first_name is not None:
                                name = f"{first_name.text} {name}"
                            authors.append(name)
                    
                    # Extract abstract
                    abstract_elem = article.find('.//AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                    
                    # Extract PMID
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    # Extract publication date
                    date_elem = article.find('.//PubDate/Year')
                    pub_year = date_elem.text if date_elem is not None else "2024"
                    
                    # Create paper dictionary
                    paper = {
                        'title': title.strip(),
                        'authors': authors,
                        'abstract': abstract.strip() if abstract else "No abstract available",
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        'published_date': f"{pub_year}-01-01T00:00:00Z",  # Simplified date
                        'pmid': pmid,
                        'source': 'PubMed',
                        'pdf_url': f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmid}/pdf/"  # May not always work
                    }
                    
                    # Apply date filter
                    if self._is_recent_enough(paper['published_date']):
                        papers.append(paper)
                
                except Exception as e:
                    logger.warning(f"Error parsing PubMed article: {e}")
                    continue
            
            logger.info(f"✅ Found {len(papers)} relevant papers from PubMed")
            
        except Exception as e:
            logger.error(f"❌ Error searching PubMed: {e}")
        
        return papers
    
    def _determine_search_sources(self, query: str) -> List[str]:
        """
        Determine which sources to search based on the query content
        
        Args:
            query (str): Research query
            
        Returns:
            List[str]: List of sources to search
        """
        query_lower = query.lower()
        sources = []
        
        # Check for CS/AI/ML keywords (expanded list)
        cs_keywords = [
            'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
            'computer vision', 'natural language processing', 'nlp', 'transformer',
            'algorithm', 'optimization', 'robotics', 'computer science', 'llm',
            'large language model', 'gpt', 'bert', 'attention mechanism', 'neural',
            'federated learning', 'reinforcement learning', 'generative model',
            'language model', 'pre-training', 'fine-tuning', 'embedding', 'reasoning'
        ]
        
        # Check for biomedical keywords (expanded list)
        bio_keywords = [
            'medicine', 'medical', 'clinical', 'disease', 'treatment', 'therapy',
            'drug', 'protein', 'gene', 'cancer', 'biomedical', 'health',
            'oncology', 'tumor', 'diagnosis', 'patient', 'clinical trial',
            'pharmaceutical', 'epidemiology', 'immunology', 'genomics', 'biotechnology'
        ]
        
        # Determine sources based on query content
        if any(keyword in query_lower for keyword in cs_keywords):
            sources.append('arxiv')
        
        if any(keyword in query_lower for keyword in bio_keywords):
            sources.append('pubmed')
        
        # If no specific match, search both for comprehensive results
        if not sources:
            sources = ['arxiv', 'pubmed']
        
        return sources
    
    def _determine_search_sources_with_router(self, query: str, keywords_data: Dict[str, Any]) -> List[str]:
        """
        Determine which sources to search using router keyword extraction
        
        Args:
            query (str): Research query  
            keywords_data (Dict): Extracted keywords from router
            
        Returns:
            List[str]: List of sources to search
        """
        # First try to use router's suggested sources
        suggested_sources = keywords_data.get('suggested_sources', [])
        
        if suggested_sources and isinstance(suggested_sources, list):
            # Convert to our internal format
            sources = []
            for source in suggested_sources:
                if source.lower() == 'arxiv':
                    sources.append('arxiv')
                elif source.lower() == 'pubmed':
                    sources.append('pubmed')
                elif source.lower() == 'both':
                    sources.extend(['arxiv', 'pubmed'])
            
            if sources:
                logger.info(f"🎯 Using router-suggested sources: {sources}")
                return list(set(sources))  # Remove duplicates
        
        # Fallback to original logic
        return self._determine_search_sources(query)
    
    def collect_papers(self, query: str, max_papers: int = 10, process_full_content: bool = True) -> List[Dict[str, Any]]:
        """
        Main method: Collect papers from multiple sources with intelligent keyword extraction
        
        Args:
            query (str): Research query
            max_papers (int): Total number of papers to return
            process_full_content (bool): Whether to extract full paper content
            
        Returns:
            List[Dict]: Combined and deduplicated list of papers
        """
        logger.info(f"🎯 Starting paper collection for: '{query}'")
        
        # Use router for intelligent keyword extraction if available
        if self.router:
            try:
                logger.info("🧠 Extracting keywords using router intelligence...")
                keywords_data = self.router.extract_academic_keywords(query)
                
                # Use optimized query for search
                optimized_query = keywords_data.get('optimized_query', query)
                logger.info(f"📊 Optimized query: '{optimized_query}'")
                
                # Determine sources using router suggestions
                sources = self._determine_search_sources_with_router(query, keywords_data)
                
                # Log extracted keywords for debugging
                primary_keywords = keywords_data.get('primary_keywords', [])
                logger.info(f"🔑 Primary keywords: {primary_keywords[:5]}")
                
                # Use optimized query for searches
                search_query = optimized_query
                
            except Exception as e:
                logger.error(f"❌ Router keyword extraction failed: {e}")
                logger.info("📋 Falling back to original query processing")
                search_query = query
                sources = self._determine_search_sources(query)
        else:
            # Fallback to original logic without router
            search_query = query
            sources = self._determine_search_sources(query)
        
        logger.info(f"📊 Will search sources: {', '.join(sources)}")
        logger.info(f"🔍 Using search query: '{search_query}'")
        
        all_papers = []
        papers_per_source = max(12, max_papers * 2)  # At least 12 per source, or 2x what user wants
        
        # Search each source with optimized query
        if 'arxiv' in sources:
            arxiv_papers = self.search_arxiv(search_query, papers_per_source)
            all_papers.extend(arxiv_papers)
        
        if 'pubmed' in sources:
            pubmed_papers = self.search_pubmed(search_query, papers_per_source)
            all_papers.extend(pubmed_papers)
        
        # Deduplicate papers by title similarity
        unique_papers = self._deduplicate_papers(all_papers)
        
        # Sort by relevance (arXiv first since it's usually more relevant for CS queries)
        unique_papers.sort(key=lambda p: (
            0 if p['source'] == 'arXiv' else 1,  # arXiv papers first
            -len(p['title'])  # Longer titles often more specific
        ))
        
        # Return top N papers
        final_papers = unique_papers[:max_papers]
        
        # Process full content if enabled
        if process_full_content and self.enable_full_papers and self.pdf_processor:
            logger.info("📄 Processing full paper content...")
            final_papers = self.process_full_papers(final_papers)
        
        logger.info(f"🏆 Collected {len(final_papers)} papers from {len(sources)} sources")
        
        return final_papers
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate papers based on title similarity
        
        Args:
            papers (List[Dict]): List of papers to deduplicate
            
        Returns:
            List[Dict]: Deduplicated list of papers
        """
        if not papers:
            return papers
        
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper['title'].lower().strip()
            
            # Simple deduplication: check if title is very similar to any seen title
            is_duplicate = False
            for seen_title in seen_titles:
                if self._titles_similar(title, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.add(title)
        
        logger.info(f"📋 Deduplicated {len(papers)} → {len(unique_papers)} papers")
        
        return unique_papers
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.85) -> bool:
        """
        Check if two titles are similar enough to be considered duplicates
        
        Args:
            title1 (str): First title
            title2 (str): Second title
            threshold (float): Similarity threshold
            
        Returns:
            bool: True if titles are similar
        """
        # Simple Jaccard similarity on words
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def get_paper_by_id(self, paper_id: str, source: str = 'arxiv') -> Optional[Dict[str, Any]]:
        """
        Fetch a specific paper by its ID
        
        Args:
            paper_id (str): Paper ID (arXiv ID or PMID)
            source (str): Source of the paper
            
        Returns:
            Optional[Dict]: Paper dictionary or None if not found
        """
        try:
            if source.lower() == 'arxiv':
                search = arxiv.Search(id_list=[paper_id])
                for result in self.arxiv_client.results(search):
                    return {
                        'title': result.title.strip(),
                        'authors': [author.name for author in result.authors],
                        'abstract': result.summary.strip(),
                        'url': result.entry_id,
                        'published_date': result.published.isoformat(),
                        'arxiv_id': result.get_short_id(),
                        'source': 'arXiv'
                    }
            
            elif source.lower() == 'pubmed':
                # Implement PubMed ID lookup if needed
                pass
                
        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
        
        return None
    
    def process_full_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process papers to extract full content when available
        
        Args:
            papers (List[Dict]): List of paper dictionaries
            
        Returns:
            List[Dict]: Papers enhanced with full content where available
        """
        if not self.pdf_processor:
            logger.warning("PDF processor not available, skipping full content extraction")
            return papers
        
        logger.info(f"📄 Processing {len(papers)} papers for full content...")
        
        enhanced_papers = []
        success_count = 0
        
        for i, paper in enumerate(papers):
            try:
                # Process the paper to get full content
                enhanced_paper = self.pdf_processor.process_paper(paper.copy())
                enhanced_papers.append(enhanced_paper)
                
                # Check if full content was successfully extracted
                if enhanced_paper.get('full_text_available', False):
                    success_count += 1
                    title = enhanced_paper.get('title', 'Unknown')[:50]
                    logger.info(f"   ✅ Extracted full content: {title}...")
                else:
                    title = enhanced_paper.get('title', 'Unknown')[:50]
                    error = enhanced_paper.get('extraction_error', 'Unknown error')
                    logger.info(f"   📋 Using abstract only: {title}... ({error})")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing paper {i+1}: {e}")
                # Keep original paper if processing fails
                enhanced_papers.append(paper)
        
        success_rate = (success_count / len(papers)) * 100 if papers else 0
        logger.info(f"📊 Full content extraction: {success_count}/{len(papers)} papers ({success_rate:.1f}%)")
        
        return enhanced_papers 