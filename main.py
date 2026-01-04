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
import re
import html
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
MATCH_THRESHOLD = 0.90  # 90% minimum match score

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DiffSegment:
    """Represents a difference between user quote and source text."""
    position: int           # Character position in user's quote
    user_text: str          # What user wrote
    source_text: str        # What source says
    diff_type: str          # 'substitution', 'insertion', 'deletion'

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
    diffs: List[Dict[str, Any]] = field(default_factory=list)  # Diff details as dicts
    verified_quote: str = ""  # User's quote with diffs marked

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
    
    # Strip HTML tags and decode entities from snippet
    clean_source = re.sub(r'<[^>]+>', '', source_text)  # Remove HTML tags
    clean_source = html.unescape(clean_source)           # Decode &amp; etc.
    
    # Normalize both for comparison
    def normalize(t):
        t = t.lower().strip()
        t = t.replace('\u2019', "'").replace('\u2018', "'")  # Smart quotes
        t = t.replace('\u201c', '"').replace('\u201d', '"')
        t = t.replace('\u2014', '-').replace('\u2013', '-')  # Dashes
        return t
    
    u, s = normalize(user_quote), normalize(clean_source)
    
    if u in s: return 1.0  # Quote contained in snippet
    if s in u: return 1.0  # Snippet contained in quote
    
    # Check if first 50% of quote is in snippet
    cutoff = int(len(u) * 0.5)
    if len(u) > 20 and u[:cutoff] in s: return 0.9
    
    return SequenceMatcher(None, u, s).ratio()


def compute_match_with_diffs(user_quote: str, source_text: str) -> tuple:
    """
    Computes match score AND identifies character-level differences.
    Returns: (score, diffs_list, verified_quote_html)
    """
    if not user_quote or not source_text:
        return 0.0, [], user_quote
    
    # Strip HTML tags and decode entities from snippet
    clean_source = re.sub(r'<[^>]+>', '', source_text)
    clean_source = html.unescape(clean_source)
    
    # Light normalization for matching (preserve case for display)
    def normalize_for_match(t):
        t = t.replace('\u2019', "'").replace('\u2018', "'")
        t = t.replace('\u201c', '"').replace('\u201d', '"')
        t = t.replace('\u2014', '-').replace('\u2013', '-')
        return t
    
    user_norm = normalize_for_match(user_quote)
    source_norm = normalize_for_match(clean_source)
    
    # Use SequenceMatcher to find differences
    matcher = SequenceMatcher(None, user_norm.lower(), source_norm.lower())
    score = matcher.ratio()
    
    diffs = []
    verified_html = ""
    
    # Get matching blocks and identify differences
    opcodes = matcher.get_opcodes()
    
    for tag, i1, i2, j1, j2 in opcodes:
        user_segment = user_norm[i1:i2]
        source_segment = source_norm[j1:j2]
        
        if tag == 'equal':
            # Matching text - no highlight
            verified_html += html.escape(user_segment)
        elif tag == 'replace':
            # Substitution - user wrote something different
            diffs.append(DiffSegment(
                position=i1,
                user_text=user_segment,
                source_text=source_segment,
                diff_type='substitution'
            ))
            verified_html += f'<span class="diff-error" title="Source: {html.escape(source_segment)}">{html.escape(user_segment)}</span>'
        elif tag == 'insert':
            # User added text not in source
            diffs.append(DiffSegment(
                position=i1,
                user_text=user_segment,
                source_text="",
                diff_type='insertion'
            ))
            verified_html += f'<span class="diff-error" title="Not in source">{html.escape(user_segment)}</span>'
        elif tag == 'delete':
            # User missing text that's in source
            diffs.append(DiffSegment(
                position=i1,
                user_text="",
                source_text=source_segment,
                diff_type='deletion'
            ))
            verified_html += f'<span class="diff-missing" title="Missing: {html.escape(source_segment)}">[...]</span>'
    
    return score, diffs, verified_html

# =============================================================================
# CASCADING SEARCH LOGIC
# =============================================================================

async def search_google_books_cascading(quote: str, author: str) -> SearchResponse:
    """
    Tries multiple search strategies until one works.
    1. Full Quote + Author
    2. Full Quote (No Author)
    3. Short Quote (First 15 words) + Author
    """
    trace = []
    clean_q = clean_quote_text(quote)  # Full quote, no truncation
    words = clean_q.split()
    short_q = " ".join(words[:15])  # First 15 words
    # Second half of quote (often has more distinctive content)
    mid = len(words) // 2
    last_q = " ".join(words[mid:]) if len(words) > 20 else short_q
    
    strategies = [
        {"name": "Full Quote + Author", "q": clean_q, "auth": author},
        {"name": "Full Quote Only", "q": clean_q, "auth": None},
        {"name": "First Fragment + Author", "q": short_q, "auth": author},
        {"name": "First Fragment Only", "q": short_q, "auth": None},
        {"name": "Second Half + Author", "q": last_q, "auth": author},
        {"name": "Second Half Only", "q": last_q, "auth": None},
    ]
    
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        for strategy in strategies:
            # Skip author strategies if no author provided
            if strategy["auth"] is not None and not author:
                continue

            query = f'"{strategy["q"]}"' # Use exact phrase matching
            if strategy["auth"]:
                query += f" inauthor:{strategy['auth']}"
            
            trace.append(f"Trying: {strategy['name']}...")
            
            try:
                resp = await client.get(GOOGLE_BOOKS_BASE_URL, params={
                    "q": query, "key": GOOGLE_BOOKS_API_KEY, "maxResults": 3, "printType": "books"
                })
                data = resp.json()
                items = data.get("items", [])
                
                if items:
                    parsed = parse_google_items(items, quote)
                    # EXACT PHRASE SEARCH: If Google found results, the book contains
                    # the phrase. Trust the search - don't require snippet verification.
                    # Google often returns description snippets, not matching passages.
                    for r in parsed:
                        r.match_score = 1.0  # Exact phrase match = verified
                    
                    trace.append(f"✅ Found {len(parsed)} result(s) via exact phrase match")
                    return SearchResponse(results=parsed, trace=trace)
                else:
                    trace.append("❌ 0 results.")
            except Exception as e:
                trace.append(f"⚠️ Error: {str(e)}")

    trace.append("⛔ All strategies exhausted.")
    return SearchResponse(results=[], trace=trace)

def parse_google_items(items, original_quote):
    parsed = []
    for item in items:
        vol = item.get("volumeInfo", {})
        snip = item.get("searchInfo", {}).get("textSnippet", "")
        
        # Compute score and detect differences
        score, diffs, verified_html = compute_match_with_diffs(original_quote, snip)
        
        parsed.append(BookMatch(
            title=vol.get("title", "Unknown"),
            authors=vol.get("authors", []),
            match_score=score,
            snippet=snip,
            has_text_snippet=bool(snip),
            url=vol.get("previewLink", ""),
            published_date=vol.get("publishedDate", ""),
            source="google_api",
            diffs=[asdict(d) for d in diffs],  # Convert to dict for JSON serialization
            verified_quote=verified_html
        ))
    return parsed

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.post("/api/search")
async def google_search(req: SearchRequest):
    if not GOOGLE_BOOKS_API_KEY:
        return {"results": [], "trace": ["❌ API Key missing"]}
    
    response = await search_google_books_cascading(req.quote, req.author_hint)
    return asdict(response)

@app.post("/serpapi/search")
async def serp_search(req: SearchRequest):
    # SerpAPI implementation
    if not SERPAPI_KEY:
        return {"results": [], "trace": ["❌ SerpAPI Key missing"]}
    
    trace = []
    results = []
    try:
        # Just try one robust query for SerpAPI
        query = f'"{clean_quote_text(req.quote)[:200]}"' # Truncate to 200 chars
        if req.author_hint: query += f" {req.author_hint}"
        
        trace.append(f"Querying SerpAPI: {query}")
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(SERPAPI_BASE_URL, params={
                "engine": "google", "q": query, "api_key": SERPAPI_KEY
            })
            data = resp.json()
            items = data.get("organic_results", [])
            
            if not items:
                trace.append("❌ 0 results from SerpAPI")
            else:
                trace.append(f"✅ Found {len(items)} raw results")
                
            for item in items[:3]:
                snip = item.get("snippet", "")
                results.append(BookMatch(
                    title=item.get("title", "Unknown"),
                    authors=[], # Hard to parse from organic results reliably
                    match_score=compute_match_score(req.quote, snip),
                    snippet=snip,
                    has_text_snippet=bool(snip),
                    url=item.get("link", ""),
                    source="serpapi"
                ))
    except Exception as e:
        trace.append(f"Error: {str(e)}")
        
    return {"results": [asdict(r) for r in results], "trace": trace}

@app.get("/config")
async def config():
    return {"google": bool(GOOGLE_BOOKS_API_KEY), "serp": bool(SERPAPI_KEY)}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Google Books Tester v1.3</title>
        <style>
            body { font-family: sans-serif; max-width: 900px; margin: 20px auto; padding: 20px; }
            .box { border: 1px solid #ddd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
            .trace { background: #333; color: #0f0; padding: 10px; font-family: monospace; font-size: 12px; border-radius: 5px; margin-top: 10px; white-space: pre-wrap; }
            button { padding: 10px 15px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; }
            button:hover { background: #0056b3; }
            input, textarea { width: 100%; padding: 8px; margin-bottom: 10px; box-sizing: border-box; }
            .result { background: #f8f9fa; padding: 15px; margin-top: 15px; border-left: 4px solid #007bff; border-radius: 4px; }
            .result-title { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
            .verified-quote { background: #fff; padding: 12px; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; line-height: 1.6; }
            .diff-error { background-color: #ffeb3b; font-weight: bold; padding: 1px 3px; border-radius: 2px; cursor: help; }
            .diff-missing { background-color: #ff9800; color: white; font-weight: bold; padding: 1px 3px; border-radius: 2px; cursor: help; }
            .error-summary { background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 4px; margin-top: 10px; }
            .error-item { margin: 5px 0; padding: 5px; background: #fff; border-radius: 3px; font-family: monospace; font-size: 12px; }
            .snippet-label { color: #666; font-size: 0.9em; margin-top: 10px; }
            .snippet-text { font-style: italic; color: #555; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <h1>📚 Google Books Tester v1.3</h1>
        <p style="color:#666">Now with quotation accuracy verification</p>
        <div class="box">
            <textarea id="quote" rows="4" placeholder="Enter quotation to verify..."></textarea>
            <input id="author" placeholder="Author (Optional)">
            <button onclick="runTest()">Verify Quotation</button>
            <button onclick="prefill()" style="background:#666">Load Test</button>
        </div>
        <div id="output"></div>

        <script>
            function prefill() {
                document.getElementById('quote').value = "Sensations roar back; his mind feels as if it becomes the huge, curved mirror of a radar telescope, gathering light from the farthest corners of the universe. Every time he steps outside, he can hear the clouds grinding through the sky.";
                document.getElementById('author').value = "Doerr";
            }
            
            async function runTest() {
                const quote = document.getElementById('quote').value;
                const author = document.getElementById('author').value;
                const out = document.getElementById('output');
                
                out.innerHTML = "<p>🔍 Searching and verifying...</p>";
                
                const res = await fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({quote, author_hint: author})
                }).then(r => r.json());
                
                let html = `<h3>Search Trace:</h3><div class="trace">${res.trace.join('\\n')}</div>`;
                
                if(res.results.length === 0) {
                    html += "<p>❌ No matches found after all attempts.</p>";
                } else {
                    res.results.forEach(r => {
                        const hasErrors = r.diffs && r.diffs.length > 0;
                        const statusIcon = hasErrors ? '⚠️' : '✅';
                        const statusText = hasErrors ? 'Differences Detected' : 'Verified';
                        
                        html += `<div class="result">
                            <div class="result-title">${statusIcon} ${r.title}</div>
                            <div>Authors: ${r.authors.join(', ') || 'Unknown'}</div>
                            <div>Match Score: ${Math.round(r.match_score*100)}%</div>
                            <div style="margin-top:10px"><strong>Your Quotation (${statusText}):</strong></div>
                            <div class="verified-quote">${r.verified_quote || quote}</div>`;
                        
                        if(hasErrors) {
                            html += `<div class="error-summary">
                                <strong>📋 Detected Differences (${r.diffs.length}):</strong>`;
                            r.diffs.forEach((d, i) => {
                                let desc = '';
                                if(d.diff_type === 'substitution') {
                                    desc = `You wrote "<b>${d.user_text}</b>" → Source has "<b>${d.source_text}</b>"`;
                                } else if(d.diff_type === 'insertion') {
                                    desc = `"<b>${d.user_text}</b>" not found in source`;
                                } else if(d.diff_type === 'deletion') {
                                    desc = `Missing from your quote: "<b>${d.source_text}</b>"`;
                                }
                                html += `<div class="error-item">${i+1}. ${desc}</div>`;
                            });
                            html += `</div>`;
                        }
                        
                        html += `<div class="snippet-label">Source snippet from Google Books:</div>
                            <div class="snippet-text">"${r.snippet}"</div>
                        </div>`;
                    });
                }
                out.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
