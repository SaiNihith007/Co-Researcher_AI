"""
PDF Processor Module for Co-Researcher.AI
=========================================

This module handles downloading and text extraction from academic papers.
It provides comprehensive text extraction from PDFs with fallback methods
and intelligent content parsing.

Features:
- Download PDFs from arXiv, PubMed, and direct URLs
- Multiple text extraction methods (PyPDF2, pdfplumber, pypdf)
- Intelligent content parsing and cleaning
- Section detection and structure preservation
- Error handling and fallback mechanisms

Author: Co-Researcher.AI Team
"""

import os
import requests
import logging
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse
import time
import re
from pathlib import Path

# PDF processing libraries with fallbacks
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Comprehensive PDF processing for academic papers
    
    This class handles the complete pipeline from URL to structured text:
    1. Download PDFs from various academic sources
    2. Extract text using multiple methods for reliability
    3. Parse and structure the content for analysis
    4. Clean and enhance text quality
    """
    
    def __init__(self, cache_dir: str = "./outputs/papers"):
        """
        Initialize the PDF processor
        
        Args:
            cache_dir (str): Directory to cache downloaded papers
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check available PDF libraries
        self.available_extractors = []
        if PDFPLUMBER_AVAILABLE:
            self.available_extractors.append("pdfplumber")
        if PYPDF2_AVAILABLE:
            self.available_extractors.append("pypdf2")
        if PYPDF_AVAILABLE:
            self.available_extractors.append("pypdf")
            
        if not self.available_extractors:
            raise RuntimeError("No PDF processing libraries available. Install PyPDF2, pdfplumber, or pypdf.")
        
        logger.info(f"📄 PDF Processor initialized with extractors: {', '.join(self.available_extractors)}")
        logger.info(f"💾 Paper cache directory: {self.cache_dir}")
    
    def get_pdf_url(self, paper: Dict[str, Any]) -> Optional[str]:
        """
        Get the PDF URL for a paper from various sources
        
        Args:
            paper (Dict): Paper dictionary with metadata
            
        Returns:
            Optional[str]: PDF URL or None if not available
        """
        # Check if PDF URL is already provided
        if paper.get('pdf_url'):
            return paper['pdf_url']
        
        # Extract from main URL based on source
        source = paper.get('source', '').lower()
        url = paper.get('url', '')
        
        if 'arxiv' in source and url:
            # Convert arXiv abstract URL to PDF URL
            if '/abs/' in url:
                pdf_url = url.replace('/abs/', '/pdf/') + '.pdf'
                return pdf_url
            elif 'arxiv.org' in url and not url.endswith('.pdf'):
                # Handle direct arXiv IDs
                arxiv_id = url.split('/')[-1]
                return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        elif 'pubmed' in source and url:
            # PubMed papers are trickier - try PMC if available
            pmid = paper.get('pmid', '')
            if pmid:
                # Try PMC free full text (may not always work)
                return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmid}/pdf/"
        
        # Try to extract PDF URL from the original URL
        if url and url.endswith('.pdf'):
            return url
        
        return None
    
    def download_pdf(self, paper: Dict[str, Any]) -> Optional[str]:
        """
        Download PDF for a paper
        
        Args:
            paper (Dict): Paper dictionary with metadata
            
        Returns:
            Optional[str]: Path to downloaded PDF or None if failed
        """
        pdf_url = self.get_pdf_url(paper)
        if not pdf_url:
            logger.warning(f"No PDF URL found for: {paper.get('title', 'Unknown')[:50]}...")
            return None
        
        # Create safe filename
        title = paper.get('title', 'unknown_paper')
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        
        # Add paper ID or hash for uniqueness
        paper_id = paper.get('arxiv_id', paper.get('pmid', ''))
        if not paper_id:
            paper_id = str(abs(hash(paper.get('url', ''))))[:8]
        
        filename = f"{safe_title}_{paper_id}.pdf"
        filepath = self.cache_dir / filename
        
        # Check if already downloaded
        if filepath.exists():
            logger.info(f"📄 PDF already cached: {filename}")
            return str(filepath)
        
        # Download the PDF
        try:
            logger.info(f"⬇️ Downloading PDF: {pdf_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; Co-Researcher.AI/1.0; research bot)'
            }
            
            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check if response is actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not pdf_url.endswith('.pdf'):
                logger.warning(f"Response may not be PDF: {content_type}")
            
            # Save to file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ PDF downloaded: {filename} ({len(response.content)} bytes)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Failed to download PDF from {pdf_url}: {e}")
            return None
    
    def extract_text_pdfplumber(self, pdf_path: str) -> Optional[str]:
        """Extract text using pdfplumber (most reliable for academic papers)"""
        try:
            import pdfplumber
            
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            text_content.append(f"\n--- Page {page_num + 1} ---\n{text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            full_text = '\n'.join(text_content)
            logger.info(f"✅ pdfplumber extracted {len(full_text)} characters")
            return full_text if full_text.strip() else None
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return None
    
    def extract_text_pypdf2(self, pdf_path: str) -> Optional[str]:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            import PyPDF2
            
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            text_content.append(f"\n--- Page {page_num + 1} ---\n{text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            full_text = '\n'.join(text_content)
            logger.info(f"✅ PyPDF2 extracted {len(full_text)} characters")
            return full_text if full_text.strip() else None
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return None
    
    def extract_text_pypdf(self, pdf_path: str) -> Optional[str]:
        """Extract text using pypdf (modern alternative)"""
        try:
            import pypdf
            
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            text_content.append(f"\n--- Page {page_num + 1} ---\n{text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            full_text = '\n'.join(text_content)
            logger.info(f"✅ pypdf extracted {len(full_text)} characters")
            return full_text if full_text.strip() else None
            
        except Exception as e:
            logger.error(f"pypdf extraction failed: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF using multiple methods with fallbacks
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            Optional[str]: Extracted text or None if all methods fail
        """
        logger.info(f"📄 Extracting text from: {Path(pdf_path).name}")
        
        # Try each extraction method in order of reliability
        extraction_methods = [
            ("pdfplumber", self.extract_text_pdfplumber),
            ("pypdf2", self.extract_text_pypdf2),
            ("pypdf", self.extract_text_pypdf)
        ]
        
        for method_name, method in extraction_methods:
            if method_name in self.available_extractors:
                logger.info(f"🔧 Trying {method_name}...")
                text = method(pdf_path)
                if text and len(text.strip()) > 100:  # Require substantial content
                    logger.info(f"✅ Successfully extracted text using {method_name}")
                    return text
                else:
                    logger.warning(f"⚠️ {method_name} extracted insufficient content")
        
        logger.error(f"❌ All extraction methods failed for {Path(pdf_path).name}")
        return None
    
    def clean_extracted_text(self, text: str) -> str:
        """
        Clean and enhance extracted PDF text
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned and structured text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r' +', ' ', text)  # Remove multiple spaces
        text = re.sub(r'\t+', ' ', text)  # Replace tabs with spaces
        
        # Remove page headers/footers patterns (common in academic papers)
        text = re.sub(r'\n--- Page \d+ ---\n', '\n\n', text)
        text = re.sub(r'\n\d+\n', '\n', text)  # Remove lone page numbers
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after sentences
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3 or not line:  # Keep substantial lines or empty lines
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)  # Max 2 consecutive newlines
        
        return cleaned_text.strip()
    
    def extract_paper_sections(self, text: str) -> Dict[str, str]:
        """
        Extract major sections from academic paper text
        
        Args:
            text (str): Full paper text
            
        Returns:
            Dict[str, str]: Dictionary with extracted sections
        """
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'discussion': '',
            'conclusion': '',
            'references': '',
            'full_text': text
        }
        
        # Common section patterns in academic papers
        section_patterns = {
            'abstract': r'(?i)\n\s*(abstract|summary)\s*\n',
            'introduction': r'(?i)\n\s*(introduction|1\.\s*introduction)\s*\n',
            'methodology': r'(?i)\n\s*(methods?|methodology|2\.\s*methods?|approach)\s*\n',
            'results': r'(?i)\n\s*(results?|findings|3\.\s*results?)\s*\n',
            'discussion': r'(?i)\n\s*(discussion|4\.\s*discussion)\s*\n',
            'conclusion': r'(?i)\n\s*(conclusion|5\.\s*conclusion|conclusions)\s*\n',
            'references': r'(?i)\n\s*(references|bibliography)\s*\n'
        }
        
        try:
            # Find section boundaries
            section_positions = {}
            for section_name, pattern in section_patterns.items():
                match = re.search(pattern, text)
                if match:
                    section_positions[section_name] = match.start()
            
            # Extract sections based on positions
            sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])
            
            for i, (section_name, start_pos) in enumerate(sorted_sections):
                # Find end position (start of next section or end of text)
                if i + 1 < len(sorted_sections):
                    end_pos = sorted_sections[i + 1][1]
                else:
                    end_pos = len(text)
                
                section_text = text[start_pos:end_pos].strip()
                
                # Clean up section header
                lines = section_text.split('\n')
                if lines and re.match(section_patterns[section_name], lines[0]):
                    section_text = '\n'.join(lines[1:]).strip()
                
                sections[section_name] = section_text
                
        except Exception as e:
            logger.warning(f"Error extracting sections: {e}")
        
        return sections
    
    def process_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete pipeline: download PDF and extract structured content
        
        Args:
            paper (Dict): Paper dictionary with metadata
            
        Returns:
            Dict: Enhanced paper with full text content
        """
        enhanced_paper = paper.copy()
        
        try:
            # Step 1: Download PDF
            pdf_path = self.download_pdf(paper)
            if not pdf_path:
                logger.warning(f"Could not download PDF for: {paper.get('title', 'Unknown')[:50]}...")
                enhanced_paper['full_text_available'] = False
                enhanced_paper['extraction_error'] = "PDF download failed"
                return enhanced_paper
            
            # Step 2: Extract text
            raw_text = self.extract_text_from_pdf(pdf_path)
            if not raw_text:
                logger.warning(f"Could not extract text from: {Path(pdf_path).name}")
                enhanced_paper['full_text_available'] = False
                enhanced_paper['extraction_error'] = "Text extraction failed"
                return enhanced_paper
            
            # Step 3: Clean text
            cleaned_text = self.clean_extracted_text(raw_text)
            
            # Step 4: Extract sections
            sections = self.extract_paper_sections(cleaned_text)
            
            # Step 5: Add to paper
            enhanced_paper.update({
                'full_text_available': True,
                'full_text': sections['full_text'],
                'sections': sections,
                'text_length': len(cleaned_text),
                'pdf_path': pdf_path,
                'extraction_method': 'success'
            })
            
            logger.info(f"✅ Successfully processed paper: {len(cleaned_text)} characters extracted")
            
        except Exception as e:
            logger.error(f"❌ Error processing paper: {e}")
            enhanced_paper['full_text_available'] = False
            enhanced_paper['extraction_error'] = str(e)
        
        return enhanced_paper
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached papers"""
        pdf_files = list(self.cache_dir.glob("*.pdf"))
        total_size = sum(f.stat().st_size for f in pdf_files)
        
        return {
            'cached_papers': len(pdf_files),
            'cache_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir)
        }
    
    def clear_cache(self):
        """Clear the paper cache"""
        try:
            for pdf_file in self.cache_dir.glob("*.pdf"):
                pdf_file.unlink()
            logger.info("🗑️ Paper cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}") 