"""
Co-Researcher.AI Agents Package
===============================

This package contains specialized agents for the Co-Researcher.AI system:

- DataCollectionAgent: Searches and collects papers from arXiv and PubMed
- SummarizationAgent: Creates structured summaries of research papers
- ReportGenerator: Generates comprehensive HTML and JSON reports
- RAGChatAgent: Handles follow-up questions using vector search and LLM

Author: Co-Researcher.AI Team
"""

from .data_collection_agent import DataCollectionAgent
from .summarization_agent import SummarizationAgent
from .report_generator import ReportGenerator
from .rag_chat_agent import RAGChatAgent

__all__ = [
    'DataCollectionAgent',
    'SummarizationAgent', 
    'ReportGenerator',
    'RAGChatAgent'
] 