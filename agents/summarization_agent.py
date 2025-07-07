"""
Summarization Agent for Co-Researcher.AI
=======================================

This agent creates detailed, structured summaries of research papers using LLM analysis.
It processes full paper content when available, with intelligent fallback to abstracts.
Extracts key information including objectives, methods, findings, and implications.

Features:
- Full paper content processing with section-aware analysis
- Intelligent text chunking for large papers
- Enhanced prompt engineering for comprehensive analysis
- Fallback mechanisms for abstract-only processing
- Cost-effective API usage with smart content selection

Author: Co-Researcher.AI Team
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class SummarizationAgent:
    """
    Agent for creating comprehensive summaries of research papers
    
    This agent analyzes full paper content when available to generate detailed summaries
    with standardized sections. Falls back gracefully to abstract-only analysis when
    full content is not available.
    """
    
    def __init__(self, llm_client):
        """
        Initialize the summarization agent
        
        Args:
            llm_client: Instance of LLMClient for generating summaries
        """
        self.llm_client = llm_client
        self.max_content_length = 8000  # Maximum characters to send to LLM
        logger.info("📝 Enhanced Summarization Agent initialized with full paper support")
    
    def summarize_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of a research paper
        
        Args:
            paper (Dict): Paper dictionary with title, abstract, authors, and optionally full content
            
        Returns:
            Dict: Structured summary with multiple sections
        """
        title = paper.get('title', 'Unknown')[:60]
        has_full_text = paper.get('full_text_available', False)
        
        logger.info(f"📝 Summarizing: {title}... (Full text: {'✅' if has_full_text else '❌'})")
        
        try:
            # Prepare the content for summarization
            content, content_type, processing_info = self._prepare_paper_content(paper)
            
            # Generate the summary using LLM
            summary = self._generate_llm_summary(content, paper, content_type, processing_info)
            
            # Add metadata
            summary['summary_timestamp'] = datetime.now().isoformat()
            summary['paper_source'] = paper.get('source', 'Unknown')
            summary['full_text_processed'] = has_full_text
            summary['content_type'] = content_type
            summary['processing_info'] = processing_info
            
            logger.info(f"✅ Summary generated successfully ({'Full paper' if has_full_text else 'Abstract-only'})")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error summarizing paper: {e}")
            # Return a basic fallback summary
            return self._create_fallback_summary(paper, str(e))
    
    def _prepare_paper_content(self, paper: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
        """
        Prepare paper content for LLM analysis with intelligent content selection
        
        Args:
            paper (Dict): Paper dictionary
            
        Returns:
            tuple: (formatted_content, content_type, processing_info)
        """
        title = paper.get('title', 'No title')
        authors = paper.get('authors', [])
        abstract = paper.get('abstract', 'No abstract')
        published_date = paper.get('published_date', 'Unknown')[:10]
        
        # Format authors
        authors_str = ', '.join(authors[:5])  # Limit to first 5 authors
        if len(authors) > 5:
            authors_str += f" (+{len(authors) - 5} more)"
        
        # Base metadata
        base_content = f"""Title: {title}

Authors: {authors_str}

Published: {published_date}

Abstract: {abstract}"""
        
        # Check if full text is available
        has_full_text = paper.get('full_text_available', False)
        processing_info = {'sections_found': [], 'content_length': 0, 'truncated': False}
        
        if has_full_text and paper.get('full_text'):
            # Process full paper content
            return self._process_full_paper_content(base_content, paper, processing_info)
        else:
            # Fall back to abstract-only processing
            processing_info['content_length'] = len(base_content)
            processing_info['fallback_reason'] = paper.get('extraction_error', 'Full text not available')
            logger.info(f"📋 Using abstract-only content ({len(base_content)} chars)")
            return base_content, 'abstract_only', processing_info
    
    def _process_full_paper_content(self, base_content: str, paper: Dict[str, Any], processing_info: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
        """
        Process full paper content with intelligent section extraction and chunking
        
        Args:
            base_content (str): Base content with metadata
            paper (Dict): Paper dictionary with full text
            processing_info (Dict): Processing information to update
            
        Returns:
            tuple: (formatted_content, content_type, processing_info)
        """
        full_text = paper.get('full_text', '')
        sections = paper.get('sections', {})
        
        # Determine the best content strategy based on available sections
        if sections and any(sections.get(key, '') for key in ['abstract', 'introduction', 'methodology', 'results', 'conclusion']):
            # Use structured sections if available
            content, content_type = self._create_structured_content(base_content, sections, processing_info)
        else:
            # Use intelligent chunking of full text
            content, content_type = self._create_chunked_content(base_content, full_text, processing_info)
        
        processing_info['content_length'] = len(content)
        
        # Check if content needs truncation
        if len(content) > self.max_content_length:
            content = self._truncate_content_intelligently(content, processing_info)
        
        logger.info(f"📄 Processed full paper: {processing_info['content_length']} chars, sections: {processing_info['sections_found']}")
        
        return content, content_type, processing_info
    
    def _create_structured_content(self, base_content: str, sections: Dict[str, str], processing_info: Dict[str, Any]) -> tuple[str, str]:
        """
        Create content from structured paper sections
        
        Args:
            base_content (str): Base metadata content
            sections (Dict): Extracted paper sections
            processing_info (Dict): Processing information to update
            
        Returns:
            tuple: (formatted_content, content_type)
        """
        content_parts = [base_content]
        
        # Define section priority and maximum lengths
        section_priorities = [
            ('introduction', 1500, 'Introduction'),
            ('methodology', 2000, 'Methodology'), 
            ('results', 2000, 'Results'),
            ('conclusion', 1000, 'Conclusion'),
            ('discussion', 1500, 'Discussion')
        ]
        
        total_added = len(base_content)
        
        for section_key, max_length, display_name in section_priorities:
            section_content = sections.get(section_key, '').strip()
            
            if section_content and total_added < self.max_content_length:
                # Truncate section if needed
                available_space = self.max_content_length - total_added - 100  # Buffer
                if len(section_content) > max_length:
                    section_content = section_content[:max_length] + '...'
                if len(section_content) > available_space:
                    section_content = section_content[:available_space] + '...'
                
                formatted_section = f"\n\n=== {display_name} ===\n{section_content}"
                content_parts.append(formatted_section)
                processing_info['sections_found'].append(section_key)
                total_added += len(formatted_section)
        
        return '\n'.join(content_parts), 'structured_sections'
    
    def _create_chunked_content(self, base_content: str, full_text: str, processing_info: Dict[str, Any]) -> tuple[str, str]:
        """
        Create content using intelligent chunking of full text
        
        Args:
            base_content (str): Base metadata content
            full_text (str): Full paper text
            processing_info (Dict): Processing information to update
            
        Returns:
            tuple: (formatted_content, content_type)
        """
        # Calculate available space for full text
        available_space = self.max_content_length - len(base_content) - 200  # Buffer
        
        if len(full_text) <= available_space:
            # Full text fits within limits
            content = f"{base_content}\n\n=== Full Paper Content ===\n{full_text}"
            processing_info['sections_found'] = ['full_text']
            return content, 'full_text'
        
        # Intelligent chunking: prioritize important sections
        chunks = self._extract_important_chunks(full_text, available_space)
        
        if chunks:
            formatted_chunks = '\n\n'.join(f"=== Excerpt {i+1} ===\n{chunk}" for i, chunk in enumerate(chunks))
            content = f"{base_content}\n\n{formatted_chunks}"
            processing_info['sections_found'] = [f'chunk_{i+1}' for i in range(len(chunks))]
            processing_info['chunking_strategy'] = 'intelligent_excerpts'
            return content, 'intelligent_chunks'
        
        # Fallback: use first portion of text
        truncated_text = full_text[:available_space] + '...'
        content = f"{base_content}\n\n=== Paper Content (Truncated) ===\n{truncated_text}"
        processing_info['sections_found'] = ['truncated_text']
        processing_info['truncated'] = True
        
        return content, 'truncated_full_text'
    
    def _extract_important_chunks(self, full_text: str, max_length: int) -> List[str]:
        """
        Extract the most important chunks from full text using heuristics
        
        Args:
            full_text (str): Full paper text
            max_length (int): Maximum total length for chunks
            
        Returns:
            List[str]: List of important text chunks
        """
        # Split text into paragraphs
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        
        # Score paragraphs by importance
        scored_paragraphs = []
        
        for para in paragraphs:
            if len(para) < 50:  # Skip very short paragraphs
                continue
                
            score = 0
            para_lower = para.lower()
            
            # Boost score for important keywords
            importance_keywords = [
                'result', 'finding', 'conclusion', 'significant', 'important',
                'novel', 'approach', 'method', 'experiment', 'analysis',
                'demonstrate', 'show', 'prove', 'evidence', 'contribution'
            ]
            
            for keyword in importance_keywords:
                score += para_lower.count(keyword) * 2
            
            # Boost score for methodology and results language
            method_keywords = ['algorithm', 'technique', 'process', 'procedure', 'implementation']
            result_keywords = ['performance', 'accuracy', 'improvement', 'comparison', 'evaluation']
            
            for keyword in method_keywords + result_keywords:
                score += para_lower.count(keyword) * 1.5
            
            # Penalize for common filler content
            if any(phrase in para_lower for phrase in ['figure', 'table', 'reference', 'acknowledgment']):
                score -= 1
            
            scored_paragraphs.append((score, para))
        
        # Sort by score and select top paragraphs within length limit
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        
        selected_chunks = []
        total_length = 0
        
        for score, para in scored_paragraphs:
            if total_length + len(para) + 50 <= max_length:  # Buffer for formatting
                selected_chunks.append(para)
                total_length += len(para) + 50
            else:
                break
        
        return selected_chunks[:4]  # Limit to 4 chunks maximum
    
    def _truncate_content_intelligently(self, content: str, processing_info: Dict[str, Any]) -> str:
        """
        Intelligently truncate content while preserving important information
        
        Args:
            content (str): Content to truncate
            processing_info (Dict): Processing information to update
            
        Returns:
            str: Truncated content
        """
        if len(content) <= self.max_content_length:
            return content
        
        # Try to truncate at natural boundaries (end of sections)
        lines = content.split('\n')
        truncated_lines = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) + 1 <= self.max_content_length - 100:  # Buffer
                truncated_lines.append(line)
                current_length += len(line) + 1
            else:
                break
        
        truncated_content = '\n'.join(truncated_lines)
        
        # Add truncation notice
        if len(truncated_content) < len(content):
            truncated_content += '\n\n... [Content truncated for analysis] ...'
            processing_info['truncated'] = True
            processing_info['original_length'] = len(content)
            processing_info['truncated_length'] = len(truncated_content)
        
        return truncated_content
    
    def _generate_llm_summary(self, content: str, paper: Dict[str, Any], content_type: str, processing_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary using LLM with enhanced prompts for full paper analysis
        
        Args:
            content (str): Formatted paper content
            paper (Dict): Original paper dictionary
            content_type (str): Type of content being processed
            processing_info (Dict): Information about content processing
            
        Returns:
            Dict: Structured summary
        """
        # Create specialized prompt based on content type
        if content_type == 'abstract_only':
            prompt = self._create_abstract_only_prompt(content)
        else:
            prompt = self._create_full_paper_prompt(content, content_type, processing_info)
        
        try:
            # Get LLM response with increased token limit for full paper analysis
            response = self.llm_client.generate_response(
                prompt,
                temperature=0.2,  # Lower temperature for more factual, consistent responses
                max_tokens=2000   # Increased for comprehensive analysis of full papers
            )
            
            # Parse the structured response
            summary = self._parse_llm_response(response)
            return summary
            
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            raise
    
    def _create_full_paper_prompt(self, content: str, content_type: str, processing_info: Dict[str, Any]) -> str:
        """
        Create enhanced prompt for full paper analysis
        
        Args:
            content (str): Paper content
            content_type (str): Type of content
            processing_info (Dict): Processing information
            
        Returns:
            str: Specialized prompt for full paper analysis
        """
        sections_info = ', '.join(processing_info.get('sections_found', []))
        
        return f"""You are an expert research analyst conducting a comprehensive analysis of a full research paper. You have access to the complete paper content, not just the abstract.

Paper Content (Content type: {content_type}, Sections available: {sections_info}):
{content}

Please provide a thorough, detailed analysis based on the FULL PAPER CONTENT. Use specific details, quotes, and insights from throughout the paper, not just the abstract. Each section should be comprehensive with 5-8 sentences:

RESEARCH OBJECTIVE:
What is the main research question or goal? What specific problem does this paper address? What gap in the literature or real-world challenge motivated this research? Why is this problem important and timely? What are the specific hypotheses or research questions?

METHODOLOGY:
What specific approaches, techniques, or methods did the researchers use? Describe the experimental design, data collection methods, datasets used, or theoretical frameworks applied. What tools, algorithms, or statistical methods were employed? How was the study structured and what were the key methodological choices? What validation approaches were used?

KEY FINDINGS:
What are the most important results or discoveries? Include specific metrics, performance improvements, statistical significance, or novel insights discovered. How do these results compare to previous work or baseline methods? What patterns, trends, or relationships were identified? What evidence supports the main conclusions?

SIGNIFICANCE & IMPACT:
Why are these findings important for the field? How do they advance current knowledge or challenge existing assumptions? What are the broader implications for research, practice, or society? What new possibilities or applications do these results enable? How do the results contribute to theory or practice?

LIMITATIONS:
What are the acknowledged limitations, constraints, or areas for improvement? What assumptions were made? What factors might limit the generalizability of the results? What methodological limitations or potential biases exist? What scope limitations are discussed?

FUTURE DIRECTIONS:
What specific future research directions are suggested by the authors? What questions remain unanswered? What extensions or improvements to the current work are proposed? How might this research inspire follow-up studies? What practical next steps are recommended?

Base your analysis on the COMPLETE PAPER CONTENT provided, not just the abstract. Extract specific details, methodological approaches, experimental results, and conclusions from the full text. Provide concrete, detailed information that demonstrates deep understanding of the entire paper."""
    
    def _create_abstract_only_prompt(self, content: str) -> str:
        """
        Create prompt for abstract-only analysis (fallback mode)
        
        Args:
            content (str): Paper content (abstract only)
            
        Returns:
            str: Prompt for abstract-based analysis
        """
        return f"""You are an expert research analyst. Analyze this research paper based on the available information (abstract and metadata only - full paper content was not accessible).

Paper Information:
{content}

Please provide a thorough analysis based on the available information. Each section should be detailed with at least 4-6 sentences. Note that this analysis is based on limited information (abstract only):

RESEARCH OBJECTIVE:
Based on the abstract, what is the main research question or goal? What specific problem does this paper appear to address? What gap in the literature or real-world challenge likely motivated this research? Why does this problem seem important and timely?

METHODOLOGY:
Based on the abstract, what approaches, techniques, or methods do the researchers appear to use? What can be inferred about the experimental design, data collection methods, or theoretical frameworks? What methodological approach is suggested by the abstract?

KEY FINDINGS:
What are the most important results or discoveries mentioned in the abstract? What evidence or insights are highlighted? How do these results appear to compare to previous work? What patterns or relationships are suggested?

SIGNIFICANCE & IMPACT:
Based on the abstract, why do these findings appear important for the field? How might they advance current knowledge? What are the potential broader implications suggested? What new possibilities might these results enable?

LIMITATIONS:
What limitations can be inferred from the abstract? What assumptions might have been made? What factors could limit the generalizability? Note that detailed limitations are typically not in abstracts.

FUTURE DIRECTIONS:
What future research directions might this work suggest? What questions appear to remain unanswered based on the abstract? What extensions might be valuable? Note that abstracts rarely discuss future work in detail.

**Note: This analysis is based on abstract and metadata only, as the full paper content was not available for processing.**"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured sections
        
        Args:
            response (str): Raw LLM response
            
        Returns:
            Dict: Parsed summary sections
        """
        summary = {}
        
        # Define section headers and their keys
        sections = {
            'RESEARCH OBJECTIVE:': 'objective',
            'METHODOLOGY:': 'methodology',
            'KEY FINDINGS:': 'key_findings',
            'SIGNIFICANCE & IMPACT:': 'significance',
            'LIMITATIONS:': 'limitations',
            'FUTURE DIRECTIONS:': 'future_directions'
        }
        
        # Split response by sections
        current_section = None
        current_content = []
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if line is a section header
            section_found = False
            for header, key in sections.items():
                if header in line.upper():
                    # Save previous section
                    if current_section and current_content:
                        summary[current_section] = ' '.join(current_content).strip()
                    
                    # Start new section
                    current_section = key
                    current_content = []
                    section_found = True
                    break
            
            # Add content to current section
            if not section_found and current_section and line:
                current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            summary[current_section] = ' '.join(current_content).strip()
        
        # Ensure all sections exist with appropriate fallback content
        for header, key in sections.items():
            if key not in summary or not summary[key]:
                if 'abstract only' in response.lower():
                    summary[key] = "Information not available in abstract - full paper analysis required."
                else:
                    summary[key] = "Information not clearly specified in the available content."
        
        return summary
    
    def _create_fallback_summary(self, paper: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """
        Create a basic fallback summary when LLM fails
        
        Args:
            paper (Dict): Paper dictionary
            error_msg (str): Error message
            
        Returns:
            Dict: Basic summary structure
        """
        abstract = paper.get('abstract', '')
        title = paper.get('title', 'Unknown research')
        authors = paper.get('authors', [])
        has_full_text = paper.get('full_text_available', False)
        
        # Extract more information from available data
        author_str = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')
        
        # Create more detailed fallback based on available content
        if has_full_text and paper.get('full_text'):
            content_description = f"full paper content ({paper.get('text_length', 0):,} characters)"
        else:
            content_description = "abstract only"
        
        if abstract:
            # Try to extract key information from abstract
            abstract_sentences = abstract.split('. ')
            first_part = '. '.join(abstract_sentences[:2]) if len(abstract_sentences) > 1 else abstract[:300]
            middle_part = '. '.join(abstract_sentences[2:4]) if len(abstract_sentences) > 3 else abstract[300:600]
            last_part = '. '.join(abstract_sentences[-2:]) if len(abstract_sentences) > 2 else abstract[-300:]
        else:
            first_part = f"This research paper titled '{title}' addresses important questions in the field."
            middle_part = "The authors present their research methodology and approach to investigate the stated objectives."
            last_part = "The paper contributes to the existing body of knowledge in this research area."
        
        return {
            'objective': f"Research focus: {title}. {first_part} The authors aim to contribute to the understanding of this important research domain through systematic investigation and analysis.",
            'methodology': f"While detailed methodology information could not be extracted due to processing limitations ({error_msg}), this work by {author_str} likely employs established research methods appropriate for the field. {middle_part} The research approach would have been designed to address the stated objectives systematically.",
            'key_findings': f"{last_part} Although specific findings could not be extracted due to technical limitations, this research contributes valuable insights to the field. The work presents results that advance our understanding of the research topic and provide evidence-based conclusions.",
            'significance': f"This research by {author_str} contributes to the broader scientific knowledge base in the field. The work addresses important questions and provides insights that may inform future research directions. The findings have potential implications for both theoretical understanding and practical applications in the domain.",
            'limitations': "Specific limitations could not be identified due to processing error. However, like all research, this work likely acknowledges certain constraints, assumptions, or areas where further investigation would be beneficial. The authors would have discussed the scope and boundaries of their findings.",
            'future_directions': f"While specific future research directions could not be extracted, this work likely opens up new avenues for investigation. The research contributes to the foundation for future studies and may suggest areas where additional research would be valuable. Follow-up work could build upon the findings presented in this paper.",
            'summary_timestamp': datetime.now().isoformat(),
            'paper_source': paper.get('source', 'Unknown'),
            'full_text_processed': has_full_text,
            'content_type': content_description,
            'error': error_msg,
            'fallback_used': True
        }
    
    def batch_summarize(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Summarize multiple papers in batch with enhanced processing
        
        Args:
            papers (List[Dict]): List of paper dictionaries
            
        Returns:
            List[Dict]: List of summaries
        """
        logger.info(f"📚 Starting batch summarization of {len(papers)} papers")
        
        summaries = []
        full_text_count = 0
        
        for i, paper in enumerate(papers, 1):
            logger.info(f"Processing paper {i}/{len(papers)}")
            
            if paper.get('full_text_available', False):
                full_text_count += 1
            
            summary = self.summarize_paper(paper)
            summaries.append(summary)
            
            # Log progress
            if i % 2 == 0:  # More frequent progress for full paper processing
                logger.info(f"✅ Completed {i}/{len(papers)} summaries ({full_text_count} with full text)")
        
        logger.info(f"🏆 Batch summarization complete: {len(summaries)} summaries generated")
        logger.info(f"📄 Full paper analysis: {full_text_count}/{len(papers)} papers ({full_text_count/len(papers)*100:.1f}%)")
        
        return summaries 