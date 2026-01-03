#!/usr/bin/env python3
"""
Google Books Standalone Test App (v1.2 - The "Cascading" Fix)
=============================================================
REVISIONS v1.2:
- Implemented "Cascading Search": If exact match fails, it automatically
  retries with broader criteria (dropping author, shortening text).
- Added "Search Trace" to UI: Shows user exactly which attempts failed.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Support both naming conventions for flexibility
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

GOOGLE_BOOKS_BASE_URL = "https://www.googleapis.com/books/v1/volumes"
SERPAPI_BASE_URL = "https://serpapi.com/search"
API_TIMEOUT = 30.0

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BookMatch:
    title: str
    authors: List[str]
    match_score: float
    snippet: str
    has_text_snippet: bool
    url: str
    published_date: str = ""
    publisher: str = ""
    source: str = ""
    error: str = ""

@dataclass
class SearchResponse:
    results: List[BookMatch]
    trace: List[str] = field(default_factory=list)

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="Google Books Test App v1.2")

class SearchRequest(BaseModel):
    quote: str
    author_hint: Optional[str] = None

# =============================================================================
# LOGIC & HELPERS
# =============================================================================

def clean_quote_text(text: str) -> str:
    """Aggressive cleaning to ensure API acceptance."""
    text = text.strip().strip('"').strip('\u201C').strip('\u201D')
    text = text.replace('\u2019', "'").replace('\u2018', "'") # Apostrophes
    text = text.replace('\u2014', ' ').replace('\u2013', ' ') # Dashes to space
    text = text.encode('ascii', 'ignore').decode('ascii')     # Remove non-ascii
    return text.strip()

def compute_match_score(user_quote: str, source_text: str) -> float:
    """Robust containment check."""
    if not user_quote or not source_text: return 0.0
    u, s = user_quote.lower().strip(), source_text.lower().strip()
    
    if u in s: return 1.0  # Perfect containment
    
    # Check if first 50% of quote is in snippet
    cutoff = int(len(u) * 0.5)
    if len(u) > 20 and u[:cutoff] in s: return 0.9
    
    return SequenceMatcher(None, u, s).ratio()

# =============================================================================
# CASCADING SEARCH LOGIC
# =============================================================================

async def search_google_books_cascading(quote: str, author: str) -> SearchResponse:
    """
    Tries multiple search strategies until one works.
    1. Full Quote + Author
    2. Full Quote (No Author)
    3. Short Quote (First
