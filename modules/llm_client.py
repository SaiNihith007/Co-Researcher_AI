"""
LLM Client Module for Co-Researcher.AI
=====================================================

This module provides a unified interface to OpenAI's GPT models with:
- Rate limiting to avoid API quota issues
- Error handling and retry logic
- Cost tracking for budget management
- Response caching to reduce redundant calls

Author: Co-Researcher.AI Team
"""

import openai
import time
import json
import hashlib
import os
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified OpenAI client with rate limiting, caching, and error handling
    
    This class handles all interactions with OpenAI's API while managing:
    - API rate limits (requests per minute)
    - Response caching to avoid duplicate calls
    - Cost tracking and budget alerts
    - Automatic retry logic for transient failures
    - API key protection against corruption
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM client with configuration
        
        Args:
            api_key (str): OpenAI API key
            model (str): Model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        # Validate API key format immediately
        if not self._validate_api_key_format(api_key):
            raise ValueError(f"Invalid API key format. Expected 'sk-...' but got: {api_key[:10]}...")
        
        # Store original API key for validation
        self._original_api_key = api_key
        self.api_key = api_key
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
        # Rate limiting: Track requests to avoid hitting API limits
        self.requests_per_minute = 20  # Conservative limit for free tier
        self.request_timestamps = []   # Track recent request times
        
        # Cost tracking: Monitor API usage
        self.total_tokens_used = 0
        self.estimated_cost = 0.0
        
        # Response caching: Avoid duplicate API calls
        self.cache_dir = "./outputs/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"✅ LLM Client initialized with model: {model}")
    
    def _validate_api_key_format(self, api_key: str) -> bool:
        """
        Simple API key format validation
        
        Args:
            api_key (str): API key to validate
            
        Returns:
            bool: True if format is valid
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        # OpenAI API keys start with 'sk-'
        if not api_key.startswith('sk-'):
            return False
        
        # Check reasonable length
        if len(api_key) < 20:
            return False
        
        return True
    
    def _verify_api_key_integrity(self):
        """
        Simple API key check
        """
        if not self._validate_api_key_format(self.api_key):
            logger.error(f"🚨 API key format invalid")
            raise ValueError("Invalid API key format")
    
    def _wait_for_rate_limit(self):
        """
        Implement rate limiting by waiting if we've made too many requests recently
        
        This prevents hitting OpenAI's rate limits which could cause errors or account suspension.
        We track timestamps of recent requests and wait if needed.
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            timestamp for timestamp in self.request_timestamps 
            if current_time - timestamp < 60
        ]
        
        # If we've made too many requests in the last minute, wait
        if len(self.request_timestamps) >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                logger.info(f"⏳ Rate limit reached, waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # Record this request timestamp
        self.request_timestamps.append(current_time)
    
    def _get_cache_key(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Generate a unique cache key for a request to enable response caching
        
        Args:
            prompt (str): The input prompt
            temperature (float): Sampling temperature
            max_tokens (int): Maximum response tokens
            
        Returns:
            str: Unique hash key for this request
        """
        # Create a unique identifier based on all request parameters
        cache_input = f"{prompt}|{self.model}|{temperature}|{max_tokens}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """
        Load a response from cache if it exists
        
        Args:
            cache_key (str): Unique identifier for the cached response
            
        Returns:
            Optional[str]: Cached response or None if not found
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still valid (24 hours)
                cached_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cached_time < timedelta(hours=24):
                    logger.info("📋 Using cached response")
                    return cache_data['response']
                    
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """
        Save a response to cache for future use
        
        Args:
            cache_key (str): Unique identifier for this response
            response (str): The response to cache
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        cache_data = {
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'model': self.model
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _update_cost_tracking(self, prompt_tokens: int, completion_tokens: int):
        """
        Track API usage costs for budget management
        
        Args:
            prompt_tokens (int): Number of input tokens
            completion_tokens (int): Number of output tokens
        """
        total_tokens = prompt_tokens + completion_tokens
        self.total_tokens_used += total_tokens
        
        # Pricing for gpt-3.5-turbo (as of 2024)
        if "gpt-3.5-turbo" in self.model:
            cost_per_1k_tokens = 0.002  # $0.002 per 1K tokens
        elif "gpt-4" in self.model:
            cost_per_1k_tokens = 0.03   # $0.03 per 1K tokens
        else:
            cost_per_1k_tokens = 0.002  # Default estimate
        
        self.estimated_cost += (total_tokens / 1000) * cost_per_1k_tokens
        
        logger.info(f"💰 Tokens used: {total_tokens} | Total cost: ${self.estimated_cost:.4f}")
    
    def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.3, 
        max_tokens: int = 800,
        use_cache: bool = True
    ) -> str:
        """
        Generate a response using OpenAI's API with all safety features
        
        This is the main method for getting LLM responses. It handles:
        - Rate limiting to avoid API issues
        - Caching to reduce costs
        - Error handling and retries
        - Cost tracking
        - API key protection
        
        Args:
            prompt (str): The input prompt for the LLM
            temperature (float): Creativity level (0.0 = deterministic, 1.0 = creative)
            max_tokens (int): Maximum length of response
            use_cache (bool): Whether to use cached responses
            
        Returns:
            str: The LLM's response
            
        Raises:
            Exception: If API call fails after retries
        """
        # Verify API key integrity before each call
        self._verify_api_key_integrity()
        
        # Check cache first to avoid unnecessary API calls
        if use_cache:
            cache_key = self._get_cache_key(prompt, temperature, max_tokens)
            cached_response = self._load_from_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Apply rate limiting to respect API quotas
        self._wait_for_rate_limit()
        
        # Attempt API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Making API call (attempt {attempt + 1}/{max_retries})")
                
                # Verify API key integrity again before the actual call
                self._verify_api_key_integrity()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Extract the response text
                response_text = response.choices[0].message.content.strip()
                
                # Track usage and costs
                if hasattr(response, 'usage'):
                    self._update_cost_tracking(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )
                
                # Cache the response for future use
                if use_cache:
                    self._save_to_cache(cache_key, response_text)
                
                logger.info("✅ API call successful")
                return response_text
                
            except openai.RateLimitError as e:
                logger.warning(f"⚠️ Rate limit hit, waiting 60 seconds...")
                time.sleep(60)
                
            except openai.APIError as e:
                logger.error(f"❌ API error on attempt {attempt + 1}: {e}")
                
                # Check if this is an API key error
                if "invalid_api_key" in str(e).lower():
                    logger.error("🚨 API key validation failed!")
                    # Try to restore the original API key
                    try:
                        self._verify_api_key_integrity()
                    except ValueError as restore_error:
                        logger.error(f"❌ Failed to restore API key: {restore_error}")
                        raise Exception("API key has been corrupted and cannot be restored")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        raise Exception("Failed to get response after all retries")
    
    def classify_query(self, query: str) -> str:
        """
        Classify a query as 'general' or 'research' using rule-based checks + LLM
        
        Args:
            query (str): User's input question
            
        Returns:
            str: 'general' or 'research'
        """
        # Simple rule-based checks for obvious general queries
        query_lower = query.lower().strip()
        
        # Handle obvious conversational/greeting queries
        conversational_patterns = [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'thanks', 'thank you', 'bye', 'goodbye', 'see you',
            'how are you', 'how do you do', 'nice to meet you', 'pleased to meet you'
        ]
        
        # Check for exact matches or very short queries
        if query_lower in conversational_patterns or len(query.strip()) <= 3:
            return "general"
        
        # Check for obvious definition requests
        definition_patterns = [
            'what is', 'what are', 'define', 'explain', 'describe',
            'how does', 'why does', 'when was', 'who is', 'where is'
        ]
        
        if any(pattern in query_lower for pattern in definition_patterns):
            # Check if it's asking for recent/latest information
            if any(word in query_lower for word in ['latest', 'recent', 'new', 'current', 'state-of-the-art', 'developments', 'advances', 'progress']):
                return "research"
            else:
                return "general"
        
        # Check for obvious research indicators
        research_indicators = [
            'latest', 'recent', 'current', 'new developments', 'advances in',
            'state-of-the-art', 'compare', 'comparison', 'versus', 'vs',
            'literature review', 'survey', 'progress in', 'trends in',
            'future directions', 'research gaps', 'systematic review'
        ]
        
        if any(indicator in query_lower for indicator in research_indicators):
            return "research"
        
        # For ambiguous cases, use LLM classification
        classification_prompt = f"""You are a query classifier for a research assistant.

Classification Rules:
- GENERAL: Simple questions asking for basic definitions, explanations, or well-established concepts that don't require recent literature analysis
- RESEARCH: Questions requiring analysis of recent papers, comparisons across studies, literature reviews, or current developments

Examples:
GENERAL: "What is BERT?", "Explain transformer architecture", "How does gradient descent work?"
RESEARCH: "Latest developments in LLM reasoning", "Compare federated learning privacy approaches", "Recent advances in few-shot learning"

Query: "{query}"

Respond with exactly one word: "general" or "research"."""
        
        try:
            response = self.generate_response(
                classification_prompt, 
                temperature=0.1, 
                max_tokens=10
            )
            
            classification = response.lower().strip()
            return classification if classification in ["general", "research"] else "general"
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return "general"  # Default to general on error (safer)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics for monitoring
        
        Returns:
            Dict with usage information
        """
        return {
            "total_tokens_used": self.total_tokens_used,
            "estimated_cost": self.estimated_cost,
            "requests_in_last_minute": len([
                t for t in self.request_timestamps 
                if time.time() - t < 60
            ]),
            "model": self.model,
            "api_key_valid": self._validate_api_key_format(self.api_key)
        } 