"""
Vector Store Module for Co-Researcher.AI
========================================

This module handles vector storage and retrieval for the RAG (Retrieval-Augmented Generation) system.
It stores paper content as embeddings and enables fast similarity-based retrieval for the chatbot.

Features:
- FAISS-based vector storage for fast similarity search
- Chunking of long papers into manageable pieces
- Metadata storage for citations and source tracking
- Persistent storage to disk

Author: Co-Researcher.AI Team
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import logging
from datetime import datetime, timedelta
import shutil
import glob

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector storage and retrieval system for RAG functionality
    
    This class manages:
    - Embedding generation for paper content
    - Vector storage using FAISS for efficient similarity search
    - Chunking long documents into retrievable pieces
    - Metadata management for proper citations
    """
    
    def __init__(
        self, 
        persist_directory: str = "./outputs/vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        session_id: str = None
    ):
        """
        Initialize the vector store with optional session isolation
        
        Args:
            persist_directory (str): Base directory to save vector database
            embedding_model (str): HuggingFace model for embeddings
            chunk_size (int): Size of text chunks for embedding
            session_id (str): Optional session ID for isolation (if None, uses shared storage)
        """
        # Set up session-specific persistence directory
        self.base_persist_directory = persist_directory  # Store base directory for cleanup
        if session_id:
            self.persist_directory = os.path.join(persist_directory, f"session_{session_id}")
            self.session_id = session_id
            self.is_session_based = True
            logger.info(f"🔒 Initializing session-based vector store: {session_id}")
        else:
            self.persist_directory = persist_directory
            self.session_id = None
            self.is_session_based = False
            logger.info("🌐 Initializing shared vector store")
        
        self.chunk_size = chunk_size
        self.embedding_model_name = embedding_model
        
        # Create persistence directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"📊 Loading embedding model: {embedding_model}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"✅ Embedding model loaded (dimension: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
        
        # Initialize FAISS index
        self.index = None
        self.metadata = []  # Store metadata for each vector
        self.document_chunks = []  # Store actual text chunks
        
        # For session-based storage, start fresh (don't load existing)
        if self.is_session_based:
            # Always start with a clean vector store for each session
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info(f"🆕 Created fresh FAISS index for session: {session_id}")
        else:
            # For shared storage, try to load existing index
            self._load_index()
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                logger.info("🆕 Created new shared FAISS index")
        
        # Run cleanup when creating new sessions
        if session_id:
            self._cleanup_old_sessions()
        
        logger.info("✅ Vector store initialized successfully")
    
    @staticmethod
    def cleanup_old_sessions(base_directory: str = "./outputs/vector_db", max_age_hours: int = 6):
        """
        Static method to clean up old session folders
        
        Args:
            base_directory: Base directory containing session folders
            max_age_hours: Maximum age in hours before cleanup
        """
        logger.info(f"🧹 Starting cleanup of sessions older than {max_age_hours} hours...")
        
        try:
            if not os.path.exists(base_directory):
                logger.info("📁 Base directory doesn't exist, nothing to clean up")
                return
            
            # Find all session directories
            session_pattern = os.path.join(base_directory, "session_*")
            session_dirs = glob.glob(session_pattern)
            
            if not session_dirs:
                logger.info("📂 No session directories found")
                return
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            cleaned_count = 0
            total_size_cleaned = 0
            
            for session_dir in session_dirs:
                try:
                    # Check directory modification time
                    dir_stat = os.stat(session_dir)
                    dir_modified = datetime.fromtimestamp(dir_stat.st_mtime)
                    
                    # Check if directory is older than cutoff
                    if dir_modified < cutoff_time:
                        # Calculate directory size before deletion
                        dir_size = VectorStore._get_directory_size(session_dir)
                        
                        # Extract session ID from directory name
                        session_id = os.path.basename(session_dir).replace("session_", "")
                        
                        # Delete the directory
                        shutil.rmtree(session_dir)
                        
                        cleaned_count += 1
                        total_size_cleaned += dir_size
                        
                        logger.info(f"🗑️  Cleaned session {session_id} (age: {datetime.now() - dir_modified}, size: {dir_size/1024/1024:.2f} MB)")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Error cleaning session {session_dir}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"✅ Cleanup complete: {cleaned_count} sessions removed, {total_size_cleaned/1024/1024:.2f} MB freed")
            else:
                logger.info("✅ Cleanup complete: No old sessions found")
                
        except Exception as e:
            logger.error(f"❌ Error during session cleanup: {e}")
    
    def _cleanup_old_sessions(self):
        """Clean up old sessions when creating a new one"""
        VectorStore.cleanup_old_sessions(self.base_persist_directory, max_age_hours=6)
    
    @staticmethod
    def _get_directory_size(directory: str) -> int:
        """
        Calculate the total size of a directory in bytes
        
        Args:
            directory: Directory path
            
        Returns:
            Total size in bytes
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        except Exception as e:
            logger.warning(f"Error calculating directory size: {e}")
        return total_size
    
    def _chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """
        Split long text into smaller chunks for better retrieval
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Maximum size of each chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        # Simple chunking by sentences to maintain coherence
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Add sentence to current chunk if it fits
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                # Start new chunk if current one is getting too long
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 50]
        
        return chunks
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity with FAISS inner product
        
        Args:
            embeddings (np.ndarray): Raw embeddings
            
        Returns:
            np.ndarray: Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    
    def add_papers(self, papers: List[Dict[str, Any]], summaries: List[str] = None):
        """
        Add papers to the vector store for RAG retrieval with full content support
        
        Args:
            papers (List[Dict]): List of paper dictionaries (may include full text)
            summaries (List[str]): Optional list of paper summaries
        """
        logger.info(f"📚 Adding {len(papers)} papers to vector store...")
        
        full_text_count = 0
        
        for i, paper in enumerate(papers):
            try:
                # Prepare comprehensive paper content
                content_parts, content_type = self._prepare_paper_content_for_vectorization(paper, summaries, i)
                
                if paper.get('full_text_available', False):
                    full_text_count += 1
                
                # Combine all content
                full_content = '\n\n'.join(content_parts)
                
                # Chunk the text with appropriate strategy
                chunks = self._chunk_text_intelligently(full_content, content_type)
                
                # Generate embeddings for each chunk
                if chunks:
                    chunk_embeddings = self.embedding_model.encode(chunks)
                    normalized_embeddings = self._normalize_embeddings(chunk_embeddings)
                    
                    # Add to FAISS index
                    self.index.add(normalized_embeddings)
                    
                    # Store enhanced metadata for each chunk
                    for j, chunk in enumerate(chunks):
                        metadata = {
                            'paper_index': i,
                            'chunk_index': j,
                            'paper_title': paper.get('title', ''),
                            'paper_authors': paper.get('authors', []),
                            'paper_url': paper.get('url', ''),
                            'paper_published_date': paper.get('published_date', ''),
                            'paper_source': paper.get('source', ''),
                            'chunk_type': self._determine_chunk_type(chunk, j, len(chunks)),
                            'full_text_available': paper.get('full_text_available', False),
                            'content_type': content_type,
                            'text_length': len(chunk)
                        }
                        
                        self.metadata.append(metadata)
                        self.document_chunks.append(chunk)
                
                title = paper.get('title', 'Unknown')[:50]
                content_status = "Full paper" if paper.get('full_text_available', False) else "Abstract only"
                logger.info(f"   ✅ Added paper {i+1}: {title}... ({len(chunks)} chunks, {content_status})")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to add paper {i+1}: {e}")
        
        # Save the updated index
        self._save_index()
        
        success_rate = (full_text_count / len(papers)) * 100 if papers else 0
        logger.info(f"✅ Successfully added papers to vector store")
        logger.info(f"📄 Full content: {full_text_count}/{len(papers)} papers ({success_rate:.1f}%)")
        logger.info(f"🔢 Total vectors: {self.index.ntotal}")
    
    def _prepare_paper_content_for_vectorization(self, paper: Dict[str, Any], summaries: List[str], paper_index: int) -> tuple[List[str], str]:
        """
        Prepare comprehensive paper content for vectorization
        
        Args:
            paper (Dict): Paper dictionary
            summaries (List[str]): List of summaries
            paper_index (int): Index of current paper
            
        Returns:
            tuple: (content_parts, content_type)
        """
        content_parts = []
        
        # Basic metadata
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        authors = ', '.join(paper.get('authors', [])[:3])
        
        # Add title and metadata
        metadata_part = f"Title: {title}\nAuthors: {authors}\nAbstract: {abstract}"
        content_parts.append(metadata_part)
        
        # Check for full text content
        has_full_text = paper.get('full_text_available', False)
        
        if has_full_text:
            return self._add_full_text_content(paper, content_parts, summaries, paper_index)
        else:
            return self._add_abstract_only_content(content_parts, summaries, paper_index)
    
    def _add_full_text_content(self, paper: Dict[str, Any], content_parts: List[str], summaries: List[str], paper_index: int) -> tuple[List[str], str]:
        """
        Add full paper content for vectorization
        
        Args:
            paper (Dict): Paper with full text
            content_parts (List[str]): Existing content parts
            summaries (List[str]): Summaries list
            paper_index (int): Paper index
            
        Returns:
            tuple: (content_parts, content_type)
        """
        # Add structured sections if available
        sections = paper.get('sections', {})
        
        if sections:
            # Add major sections
            section_priorities = [
                ('introduction', 'Introduction'),
                ('methodology', 'Methods'),
                ('results', 'Results'), 
                ('discussion', 'Discussion'),
                ('conclusion', 'Conclusion')
            ]
            
            for section_key, section_name in section_priorities:
                section_content = sections.get(section_key, '').strip()
                if section_content:
                    # Limit section length for vectorization
                    if len(section_content) > 3000:
                        section_content = section_content[:3000] + '...'
                    content_parts.append(f"{section_name}:\n{section_content}")
        
        # Add full text if sections aren't available or are limited
        full_text = paper.get('full_text', '')
        if full_text and len('\n\n'.join(content_parts)) < 5000:  # Don't exceed reasonable limit
            # Add portion of full text
            remaining_space = 8000 - len('\n\n'.join(content_parts))
            if remaining_space > 1000:
                text_portion = full_text[:remaining_space] + '...' if len(full_text) > remaining_space else full_text
                content_parts.append(f"Full Paper Content:\n{text_portion}")
        
        # Add AI summary if available
        if summaries and paper_index < len(summaries):
            summary = summaries[paper_index]
            if isinstance(summary, dict):
                # Extract key sections from structured summary
                summary_parts = []
                for key in ['objective', 'methodology', 'key_findings', 'significance']:
                    value = summary.get(key, '')
                    if value and len(value) > 20:
                        summary_parts.append(f"{key.title()}: {value}")
                
                if summary_parts:
                    content_parts.append("AI Summary:\n" + '\n\n'.join(summary_parts))
            elif isinstance(summary, str) and len(summary) > 20:
                content_parts.append(f"AI Summary:\n{summary}")
        
        return content_parts, 'full_paper_content'
    
    def _add_abstract_only_content(self, content_parts: List[str], summaries: List[str], paper_index: int) -> tuple[List[str], str]:
        """
        Add abstract-only content for vectorization (fallback)
        
        Args:
            content_parts (List[str]): Existing content parts  
            summaries (List[str]): Summaries list
            paper_index (int): Paper index
            
        Returns:
            tuple: (content_parts, content_type)
        """
        # Add AI summary if available (more important for abstract-only papers)
        if summaries and paper_index < len(summaries):
            summary = summaries[paper_index]
            if isinstance(summary, dict):
                # Extract all sections from structured summary
                summary_parts = []
                for key, value in summary.items():
                    if key not in ['summary_timestamp', 'paper_source', 'error', 'fallback_used'] and isinstance(value, str) and len(value) > 20:
                        summary_parts.append(f"{key.replace('_', ' ').title()}: {value}")
                
                if summary_parts:
                    content_parts.append("Detailed AI Analysis:\n" + '\n\n'.join(summary_parts))
            elif isinstance(summary, str) and len(summary) > 20:
                content_parts.append(f"Summary:\n{summary}")
        
        return content_parts, 'abstract_with_summary'
    
    def _chunk_text_intelligently(self, text: str, content_type: str) -> List[str]:
        """
        Chunk text with strategy based on content type
        
        Args:
            text (str): Text to chunk
            content_type (str): Type of content being chunked
            
        Returns:
            List[str]: List of text chunks
        """
        if content_type == 'full_paper_content':
            # Use larger chunks for full papers to preserve context
            return self._chunk_text(text, chunk_size=800)
        else:
            # Use standard chunks for abstract-only content
            return self._chunk_text(text, chunk_size=500)
    
    def _determine_chunk_type(self, chunk: str, chunk_index: int, total_chunks: int) -> str:
        """
        Determine the type of content in a chunk for better retrieval
        
        Args:
            chunk (str): Text chunk
            chunk_index (int): Index of chunk
            total_chunks (int): Total number of chunks
            
        Returns:
            str: Chunk type classification
        """
        chunk_lower = chunk.lower()
        
        # Metadata chunk (usually first)
        if chunk_index == 0 and ('title:' in chunk_lower or 'authors:' in chunk_lower):
            return 'metadata'
        
        # Introduction/background
        if any(keyword in chunk_lower for keyword in ['introduction', 'background', 'motivation', 'related work']):
            return 'introduction'
        
        # Methodology 
        if any(keyword in chunk_lower for keyword in ['method', 'approach', 'algorithm', 'procedure', 'experiment']):
            return 'methodology'
        
        # Results
        if any(keyword in chunk_lower for keyword in ['result', 'finding', 'evaluation', 'performance', 'accuracy']):
            return 'results'
        
        # Discussion/conclusion
        if any(keyword in chunk_lower for keyword in ['discussion', 'conclusion', 'limitation', 'future work']):
            return 'discussion'
        
        # AI summary
        if 'ai summary' in chunk_lower or 'detailed ai analysis' in chunk_lower:
            return 'ai_summary'
        
        # Default
        if chunk_index < total_chunks * 0.3:
            return 'early_content'
        elif chunk_index > total_chunks * 0.7:
            return 'late_content'
        else:
            return 'main_content'
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for relevant chunks based on query
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List[Tuple[str, Dict, float]]: List of (chunk_text, metadata, similarity_score)
        """
        if self.index.ntotal == 0:
            logger.warning("No documents in vector store")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            normalized_query = self._normalize_embeddings(query_embedding)
            
            # Search FAISS index
            similarities, indices = self.index.search(normalized_query, min(top_k, self.index.ntotal))
            
            # Prepare results
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < len(self.document_chunks):  # Valid index
                    chunk_text = self.document_chunks[idx]
                    metadata = self.metadata[idx]
                    similarity_score = float(sim)
                    
                    results.append((chunk_text, metadata, similarity_score))
            
            logger.info(f"🔍 Found {len(results)} relevant chunks for query")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching vector store: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get relevant context for a query, formatted for LLM input
        
        Args:
            query (str): User's question
            max_context_length (int): Maximum length of context to return
            
        Returns:
            Tuple[str, List[Dict]]: (formatted_context, source_papers)
        """
        # Search for relevant chunks
        search_results = self.search(query, top_k=10)
        
        if not search_results:
            return "No relevant information found in the research papers.", []
        
        # Build context string and track sources
        context_parts = []
        source_papers = []
        current_length = 0
        seen_papers = set()
        
        for chunk_text, metadata, similarity_score in search_results:
            # Check if we have space for this chunk
            if current_length + len(chunk_text) > max_context_length:
                break
            
            # Add chunk to context
            paper_title = metadata['paper_title']
            context_part = f"From '{paper_title}':\n{chunk_text}\n"
            context_parts.append(context_part)
            current_length += len(context_part)
            
            # Track unique source papers
            paper_key = metadata['paper_title']
            if paper_key not in seen_papers:
                seen_papers.add(paper_key)
                source_papers.append({
                    'title': metadata['paper_title'],
                    'authors': metadata['paper_authors'],
                    'url': metadata['paper_url'],
                    'published_date': metadata['paper_published_date']
                })
        
        # Format the final context
        formatted_context = "\n---\n".join(context_parts)
        
        logger.info(f"📝 Generated context from {len(source_papers)} papers ({current_length} characters)")
        
        return formatted_context, source_papers
    
    def _save_index(self):
        """Save the FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            index_path = os.path.join(self.persist_directory, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
            
            # Save metadata and chunks
            metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'document_chunks': self.document_chunks,
                    'embedding_model': self.embedding_model_name,
                    'embedding_dim': self.embedding_dim
                }, f)
            
            logger.info("💾 Vector store saved to disk")
            
        except Exception as e:
            logger.error(f"❌ Failed to save vector store: {e}")
    
    def _load_index(self):
        """Load the FAISS index and metadata from disk"""
        try:
            index_path = os.path.join(self.persist_directory, "faiss_index.bin")
            metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load metadata and chunks
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data['metadata']
                    self.document_chunks = data['document_chunks']
                
                logger.info(f"📂 Loaded existing vector store ({self.index.ntotal} vectors)")
                
        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}")
            self.index = None
    
    def clear(self):
        """Clear all data from the vector store"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []
        self.document_chunks = []
        
        # Remove saved files
        try:
            index_path = os.path.join(self.persist_directory, "faiss_index.bin")
            metadata_path = os.path.join(self.persist_directory, "metadata.pkl")
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            logger.info("🗑️ Vector store cleared")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store including session cleanup info
        
        Returns:
            Dict: Statistics including document count, embeddings, session info, cleanup status
        """
        stats = {
            'total_documents': len(self.document_chunks),
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim,
            'chunk_size': self.chunk_size,
            'persist_directory': self.persist_directory,
            'is_session_based': self.is_session_based,
            'session_id': self.session_id
        }
        
        if self.metadata:
            # Count unique papers
            unique_papers = set(meta['paper_title'] for meta in self.metadata)
            stats['unique_papers'] = len(unique_papers)
            
            # Session isolation info
            if self.is_session_based:
                stats['session_info'] = f"Isolated session: {self.session_id}"
                stats['isolation_level'] = "SESSION_ISOLATED"
            else:
                stats['session_info'] = "Shared across all sessions"
                stats['isolation_level'] = "SHARED"
        
        # Add session cleanup information
        if self.is_session_based:
            try:
                # Count existing session directories
                session_pattern = os.path.join(self.base_persist_directory, "session_*")
                existing_sessions = glob.glob(session_pattern)
                
                # Calculate ages of existing sessions
                old_sessions = 0
                total_size = 0
                
                cutoff_time = datetime.now() - timedelta(hours=6)
                
                for session_dir in existing_sessions:
                    try:
                        dir_stat = os.stat(session_dir)
                        dir_modified = datetime.fromtimestamp(dir_stat.st_mtime)
                        
                        if dir_modified < cutoff_time:
                            old_sessions += 1
                        
                        total_size += self._get_directory_size(session_dir)
                        
                    except Exception:
                        continue
                
                stats['cleanup_info'] = {
                    'total_sessions': len(existing_sessions),
                    'old_sessions_ready_for_cleanup': old_sessions,
                    'total_storage_size_mb': round(total_size / 1024 / 1024, 2),
                    'cleanup_age_threshold_hours': 6,
                    'cleanup_enabled': True
                }
                
            except Exception as e:
                stats['cleanup_info'] = {
                    'error': f"Could not gather cleanup stats: {e}",
                    'cleanup_enabled': True
                }
        else:
            stats['cleanup_info'] = {
                'cleanup_enabled': False,
                'reason': 'Shared vector store, no session cleanup needed'
            }
        
        return stats 