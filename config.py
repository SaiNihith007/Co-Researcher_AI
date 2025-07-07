"""
Configuration for Co-Researcher.AI
Handles API keys, model settings, and other configuration
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables and defaults"""
    
    config = {
        # OpenAI Configuration
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "800"))
        },
        
        # Research Settings
        "research": {
            "max_papers": int(os.getenv("MAX_PAPERS", "10")),
            "years_back": int(os.getenv("YEARS_BACK", "3")),
            "min_relevance_score": float(os.getenv("MIN_RELEVANCE_SCORE", "0.3")),
            "sources": os.getenv("PAPER_SOURCES", "arXiv,PubMed").split(",")
        },
        
        # Embedding Settings
        "embeddings": {
            "model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
        },
        
        # Vector Database
        "vector_db": {
            "type": os.getenv("VECTOR_DB_TYPE", "faiss"),  # faiss or chroma
            "persist_directory": os.getenv("VECTOR_DB_PATH", "./outputs/vector_db")
        },
        
        # Output Settings
        "output": {
            "reports_dir": os.getenv("REPORTS_DIR", "./outputs/reports"),
            "format": os.getenv("REPORT_FORMAT", "pdf"),  # pdf or markdown
            "include_citations": bool(os.getenv("INCLUDE_CITATIONS", "true"))
        },
        
        # Rate Limiting
        "rate_limits": {
            "api_calls_per_minute": int(os.getenv("API_CALLS_PER_MINUTE", "20")),
            "concurrent_requests": int(os.getenv("CONCURRENT_REQUESTS", "3"))
        }
    }
    
    return config

def get_api_key(service: str) -> str:
    """Get API key for a specific service"""
    
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "pubmed": os.getenv("PUBMED_API_KEY"),  # Optional
        "semantic_scholar": os.getenv("SEMANTIC_SCHOLAR_API_KEY")  # Optional
    }
    
    return api_keys.get(service)

def validate_config() -> bool:
    """Validate that required configuration is present"""
    
    required_vars = [
        "OPENAI_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def setup_directories():
    """Create necessary directories if they don't exist"""
    
    config = load_config()
    
    directories = [
        config["output"]["reports_dir"],
        config["vector_db"]["persist_directory"],
        "./outputs/cache"  # For caching API responses
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Default configuration for free/low-cost usage
FREE_TIER_CONFIG = {
    "openai_model": "gpt-3.5-turbo",  # Cheapest OpenAI model
    "max_papers": 5,                   # Limit paper analysis
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # Free embeddings
    "cache_responses": True,           # Cache to avoid repeat API calls
    "rate_limit": 10                   # Conservative rate limiting
} 