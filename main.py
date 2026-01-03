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
    3. Short Quote (First 15 words) + Author
    """
    trace = []
    clean_q = clean_quote_text(quote)
    short_q = " ".join(clean_q.split()[:15]) # First 15 words
    
    strategies = [
        {"name": "Full Quote + Author", "q": clean_q, "auth": author},
        {"name": "Full Quote Only (Relaxed Author)", "q": clean_q, "auth": None},
        {"name": "Short Fragment + Author", "q": short_q, "auth": author},
    ]
    
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        for strategy in strategies:
            # Skip if strategy is invalid (e.g. no author provided)
            if strategy["name"].endswith("Author") and not author:
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
                    trace.append(f"✅ Success! Found {len(items)} results.")
                    return SearchResponse(
                        results=parse_google_items(items, quote),
                        trace=trace
                    )
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
        parsed.append(BookMatch(
            title=vol.get("title", "Unknown"),
            authors=vol.get("authors", []),
            match_score=compute_match_score(original_quote, snip),
            snippet=snip,
            has_text_snippet=bool(snip),
            url=vol.get("previewLink", ""),
            published_date=vol.get("publishedDate", ""),
            source="google_api"
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
        <title>Google Books Tester v1.2</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; }
            .box { border: 1px solid #ddd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
            .trace { background: #333; color: #0f0; padding: 10px; font-family: monospace; font-size: 12px; border-radius: 5px; margin-top: 10px; white-space: pre-wrap; }
            button { padding: 10px 15px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; }
            button:hover { background: #0056b3; }
            input, textarea { width: 100%; padding: 8px; margin-bottom: 10px; box-sizing: border-box; }
            .result { background: #f8f9fa; padding: 10px; margin-top: 10px; border-left: 4px solid #007bff; }
        </style>
    </head>
    <body>
        <h1>📚 Google Books Tester v1.2</h1>
        <div class="box">
            <textarea id="quote" placeholder="Quote..."></textarea>
            <input id="author" placeholder="Author (Optional)">
            <button onclick="runTest()">Run Diagnostics</button>
            <button onclick="prefill()" style="background:#666">Load Test</button>
        </div>
        <div id="output"></div>

        <script>
            function prefill() {
                document.getElementById('quote').value = "One for the toilet, one for the basin—Seymour stops taking the buspirone.";
                document.getElementById('author').value = "Doerr";
            }
            
            async function runTest() {
                const quote = document.getElementById('quote').value;
                const author = document.getElementById('author').value;
                const out = document.getElementById('output');
                
                out.innerHTML = "Running Cascading Search...";
                
                const res = await fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({quote, author_hint: author})
                }).then(r => r.json());
                
                let html = `<h3>Google Books API Trace:</h3><div class="trace">${res.trace.join('\\n')}</div>`;
                
                if(res.results.length === 0) {
                    html += "<p>❌ No matches found after all attempts.</p>";
                } else {
                    res.results.forEach(r => {
                        html += `<div class="result">
                            <strong>${r.title}</strong> (${Math.round(r.match_score*100)}% Match)<br>
                            <em>"${r.snippet}..."</em>
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
