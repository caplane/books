#!/usr/bin/env python3
"""
Google Books Standalone Test App (Revised)
==========================================

Isolated test environment for debugging Google Books API integration.
Tests BOTH the native Google Books API AND SerpAPI Google Books.

REVISIONS:
- Default search is now "Fuzzy" (not exact phrase) to prevent 0 results.
- Match scoring logic fixed to detect quotes INSIDE snippets.
- Security improved on /config endpoint.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "")
# Also check GOOGLE_API_KEY as fallback
if not GOOGLE_BOOKS_API_KEY:
    GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_API_KEY", "")

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

GOOGLE_BOOKS_BASE_URL = "https://www.googleapis.com/books/v1/volumes"
SERPAPI_BASE_URL = "https://serpapi.com/search"
API_TIMEOUT = 30.0

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BookMatch:
    """Result from book search."""
    success: bool
    title: str
    authors: List[str]
    publisher: str
    published_date: str
    snippet: str
    page_number: Optional[int]
    url: str
    isbn: str
    match_score: float
    source: str  # "google_api" or "serpapi"
    error: str = ""
    
    # Debug info
    has_text_snippet: bool = False
    preview_link: str = ""
    info_link: str = ""

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Google Books Test App",
    description="Standalone test for Google Books API and SerpAPI integration",
    version="1.1.0"
)

# =============================================================================
# REQUEST MODELS
# =============================================================================

class SearchRequest(BaseModel):
    quote: str
    author_hint: Optional[str] = None
    max_results: int = 5
    # REVISION: Default to False to be more forgiving with punctuation
    use_exact_phrase: bool = False 

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_match_score(user_quote: str, source_text: str) -> float:
    """
    Compute similarity between user quote and source text.
    REVISION: Checks for containment first, then falls back to ratio.
    """
    if not user_quote or not source_text:
        return 0.0

    user_clean = user_quote.lower().strip()
    source_clean = source_text.lower().strip()
    
    # 1. Exact containment (The quote is inside the snippet)
    if user_clean in source_clean:
        return 1.0
        
    # 2. Fuzzy containment (If the user quote is long, check if a major chunk exists)
    # This helps if the API cuts off the snippet halfway through the quote
    if len(user_clean) > 20:
        cutoff = int(len(user_clean) * 0.8) # Check first 80%
        if user_clean[:cutoff] in source_clean:
            return 0.95

    # 3. Fallback to sequence matching for imperfect matches
    return SequenceMatcher(None, user_clean, source_clean).ratio()


def clean_quote_text(text: str) -> str:
    """
    Clean special characters that break Google Books API search.
    """
    # Step 1: Strip semantic boundary quotes
    text = text.strip().strip('"').strip('\u201C').strip('\u201D').strip()
    
    # Step 2: Normalize quotes
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    
    # Step 3: Normalize dashes and spaces
    text = text.replace('\u2014', '-').replace('\u2013', '-')
    text = text.replace('\u00A0', ' ')
    
    # Step 4: ASCII fallback
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Step 5: Remove internal double quotes to prevent query breakage
    text = text.replace('"', '')
    
    return text.strip()

# =============================================================================
# NATIVE GOOGLE BOOKS API
# =============================================================================

async def search_google_books_api(
    quote_text: str,
    author_hint: Optional[str] = None,
    max_results: int = 5,
    use_exact_phrase: bool = False
) -> List[BookMatch]:
    """Search using native Google Books API."""
    results = []
    
    logger.info(f"--- Google API Search: '{quote_text[:30]}...' (Exact: {use_exact_phrase}) ---")
    
    if not GOOGLE_BOOKS_API_KEY:
        return [BookMatch(
            success=False, title="", authors=[], publisher="",
            published_date="", snippet="", page_number=None,
            url="", isbn="", match_score=0.0, source="google_api",
            error="GOOGLE_BOOKS_API_KEY not configured"
        )]
    
    clean_text = clean_quote_text(quote_text)
    search_text = clean_text[:200].strip()
    
    # Query Construction
    if use_exact_phrase:
        query = f'"{search_text}"'
    else:
        # For fuzzy search, we just send the text. Google handles the rest.
        query = search_text
    
    if author_hint:
        query += f" inauthor:{author_hint}"
    
    params = {
        "q": query,
        "key": GOOGLE_BOOKS_API_KEY,
        "maxResults": max_results,
        "printType": "books"
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(GOOGLE_BOOKS_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            
            # REVISION: If exact phrase failed (0 items), automatically fallback to fuzzy
            if not items and use_exact_phrase:
                logger.info("Exact phrase returned 0 results. Retrying with fuzzy search...")
                return await search_google_books_api(
                    quote_text, author_hint, max_results, use_exact_phrase=False
                )
            
            for item in items[:max_results]:
                volume_info = item.get("volumeInfo", {})
                search_info = item.get("searchInfo", {})
                
                title = volume_info.get("title", "Unknown")
                authors = volume_info.get("authors", [])
                snippet = search_info.get("textSnippet", "")
                
                # Compute match score using new logic
                match_score = compute_match_score(quote_text, snippet) if snippet else 0.0
                
                results.append(BookMatch(
                    success=True,
                    title=title,
                    authors=authors,
                    publisher=volume_info.get("publisher", ""),
                    published_date=volume_info.get("publishedDate", ""),
                    snippet=snippet,
                    page_number=None,
                    url=volume_info.get("previewLink", ""),
                    isbn="", # Simplified for brevity
                    match_score=match_score,
                    source="google_api",
                    has_text_snippet=bool(snippet),
                    preview_link=volume_info.get("previewLink", ""),
                    info_link=volume_info.get("infoLink", "")
                ))
            
            results.sort(key=lambda x: x.match_score, reverse=True)
            
    except Exception as e:
        logger.error(f"Google API Error: {e}")
        results.append(BookMatch(
            success=False, title="", authors=[], publisher="",
            published_date="", snippet="", page_number=None,
            url="", isbn="", match_score=0.0, source="google_api",
            error=str(e)
        ))
    
    return results

# =============================================================================
# SERPAPI GOOGLE BOOKS
# =============================================================================

async def search_serpapi_books(
    quote_text: str,
    author_hint: Optional[str] = None,
    max_results: int = 5
) -> List[BookMatch]:
    """Search using SerpAPI Google Books engine."""
    results = []
    
    if not SERPAPI_KEY:
        return [BookMatch(
            success=False, title="", authors=[], publisher="",
            published_date="", snippet="", page_number=None,
            url="", isbn="", match_score=0.0, source="serpapi",
            error="SERPAPI_KEY not configured"
        )]
    
    search_text = quote_text[:200].strip()
    
    # SerpAPI (Google Web Search) usually handles quotes well, 
    # but we can also relax this if needed. Staying with quotes for now.
    query = f'"{search_text}"'
    if author_hint:
        query += f" {author_hint}"
    
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        # "tbm": "bks" # Optional: force books mode if SerpAPI supports it fully
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(SERPAPI_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            items = data.get("organic_results", [])
            
            for item in items[:max_results]:
                snippet = item.get("snippet", "")
                match_score = compute_match_score(quote_text, snippet)
                
                title = item.get("title", "Unknown")
                
                # Try to parse page number from title
                page_number = None
                if " - Page " in title:
                    parts = title.rsplit(" - Page ", 1)
                    title = parts[0]
                    try:
                        page_number = int(parts[1])
                    except: pass

                results.append(BookMatch(
                    success=True,
                    title=title,
                    authors=[], # SerpAPI authors parsing is complex, skipping for cleaner code
                    publisher="",
                    published_date="",
                    snippet=snippet,
                    page_number=page_number,
                    url=item.get("link", ""),
                    isbn="",
                    match_score=match_score,
                    source="serpapi",
                    has_text_snippet=bool(snippet)
                ))
            
            results.sort(key=lambda x: x.match_score, reverse=True)
            
    except Exception as e:
        logger.error(f"SerpAPI Error: {e}")
        results.append(BookMatch(
            success=False, title="", authors=[], publisher="",
            published_date="", snippet="", page_number=None,
            url="", isbn="", match_score=0.0, source="serpapi",
            error=str(e)
        ))
    
    return results

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Web UI for testing."""
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Google Books Test App</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f9f9f9; }}
            .container {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
            h1 {{ color: #1a73e8; margin-top: 0; }}
            
            .status-bar {{ display: flex; gap: 15px; margin-bottom: 25px; }}
            .status-pill {{ flex: 1; padding: 12px; border-radius: 8px; font-size: 14px; display: flex; align-items: center; justify-content: space-between; }}
            .status-pill.ok {{ background: #e6f4ea; color: #137333; border: 1px solid #ceead6; }}
            .status-pill.error {{ background: #fce8e6; color: #c5221f; border: 1px solid #fad2cf; }}
            
            textarea {{ width: 100%; height: 100px; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-family: inherit; font-size: 15px; margin-bottom: 10px; resize: vertical; box-sizing: border-box; }}
            input[type="text"] {{ width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 15px; box-sizing: border-box; }}
            
            .controls {{ display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; }}
            button {{ padding: 12px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 14px; transition: opacity 0.2s; }}
            button:hover {{ opacity: 0.9; }}
            .btn-google {{ background: #4285f4; color: white; }}
            .btn-serp {{ background: #34a853; color: white; }}
            .btn-both {{ background: #673ab7; color: white; }}
            .btn-test {{ background: #e0e0e0; color: #333; font-weight: normal; font-size: 13px; }}
            
            .result-card {{ border: 1px solid #eee; padding: 20px; border-radius: 8px; margin-bottom: 15px; background: white; transition: transform 0.1s; }}
            .result-card:hover {{ border-color: #ccc; }}
            .result-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }}
            .match-badge {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
            .match-high {{ background: #e6f4ea; color: #137333; }}
            .match-med {{ background: #fef7e0; color: #b06000; }}
            .match-low {{ background: #fce8e6; color: #c5221f; }}
            
            .snippet-box {{ background: #f8f9fa; padding: 12px; border-left: 3px solid #ddd; margin: 10px 0; font-style: italic; color: #444; font-size: 14px; line-height: 1.5; }}
            .warning {{ color: #d93025; font-size: 13px; font-weight: 500; display: flex; align-items: center; gap: 5px; }}
            
            .raw-toggle {{ color: #666; font-size: 12px; text-decoration: underline; cursor: pointer; margin-top: 5px; display: inline-block; }}
            .raw-data {{ display: none; background: #2d2d2d; color: #ccc; padding: 15px; border-radius: 8px; font-size: 11px; overflow-x: auto; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📚 Google Books Tester (v1.1)</h1>
            
            <div class="status-bar" id="statusBar">Loading config...</div>
            
            <div>
                <label style="font-weight:bold; display:block; margin-bottom:5px;">Quote Text:</label>
                <textarea id="quote" placeholder="Paste your book quote here..."></textarea>
                
                <label style="font-weight:bold; display:block; margin-bottom:5px;">Author Hint (Optional):</label>
                <input type="text" id="author" placeholder="e.g. Orwell, Austin">
            </div>
            
            <div class="controls">
                <button class="btn-google" onclick="search('google')">🔵 Google API</button>
                <button class="btn-serp" onclick="search('serp')">🟢 SerpAPI</button>
                <button class="btn-both" onclick="search('both')">🟣 Compare Both</button>
            </div>
            
            <div style="border-top: 1px solid #eee; padding-top: 15px; margin-top: 15px;">
                <span style="font-size:12px; color:#666; margin-right: 10px;">LOAD TEST DATA:</span>
                <button class="btn-test" onclick="loadTest('prozac')">Listening to Prozac</button>
                <button class="btn-test" onclick="loadTest('einstein')">Einstein</button>
                <button class="btn-test" onclick="loadTest('potter')">Harry Potter</button>
            </div>
            
            <div id="resultsArea" style="margin-top: 30px;"></div>
        </div>
        
        <script>
            // --- CONFIG CHECK ---
            fetch('/config').then(r => r.json()).then(data => {{
                const gClass = data.google_ok ? 'ok' : 'error';
                const sClass = data.serp_ok ? 'ok' : 'error';
                document.getElementById('statusBar').innerHTML = `
                    <div class="status-pill ${{gClass}}">
                        <span>Google Books API</span>
                        <strong>${{data.google_ok ? 'Ready' : 'Missing'}}</strong>
                    </div>
                    <div class="status-pill ${{sClass}}">
                        <span>SerpAPI</span>
                        <strong>${{data.serp_ok ? 'Ready' : 'Missing'}}</strong>
                    </div>
                `;
            }});

            // --- SEARCH LOGIC ---
            async function search(type) {{
                const quote = document.getElementById('quote').value.trim();
                const author = document.getElementById('author').value.trim();
                const area = document.getElementById('resultsArea');
                
                if (!quote) {{ alert("Please enter text first!"); return; }}
                
                area.innerHTML = '<p style="text-align:center; color:#666;">Searching... ⏳</p>';
                
                const payload = {{ quote, author_hint: author, max_results: 3, use_exact_phrase: false }};
                
                try {{
                    if (type === 'both') {{
                        const [gRes, sRes] = await Promise.all([
                            doFetch('/api/search', payload),
                            doFetch('/serpapi/search', payload)
                        ]);
                        area.innerHTML = renderSection('Google API', gRes) + renderSection('SerpAPI', sRes);
                    }} else {{
                        const endpoint = type === 'google' ? '/api/search' : '/serpapi/search';
                        const res = await doFetch(endpoint, payload);
                        area.innerHTML = renderSection(type === 'google' ? 'Google API' : 'SerpAPI', res);
                    }}
                }} catch (e) {{
                    area.innerHTML = `<div class="status-pill error">Error: ${{e.message}}</div>`;
                }}
            }}
            
            async function doFetch(url, data) {{
                const r = await fetch(url, {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(data)
                }});
                return r.json();
            }}

            // --- RENDERING ---
            function renderSection(title, results) {{
                let html = `<h2 style="margin-top:30px; border-bottom:2px solid #eee; padding-bottom:10px;">${{title}} (${{results.length}})</h2>`;
                
                if (results.length === 0) return html + '<p style="color:#666;">No results found.</p>';
                
                results.forEach((r, idx) => {{
                    const scorePct = Math.round(r.match_score * 100);
                    let badgeClass = 'match-low';
                    if (scorePct > 80) badgeClass = 'match-high';
                    else if (scorePct > 40) badgeClass = 'match-med';
                    
                    html += `
                    <div class="result-card">
                        <div class="result-header">
                            <div>
                                <h3 style="margin:0; color:#1a73e8;">${{r.title}}</h3>
                                <div style="font-size:13px; color:#555; margin-top:4px;">
                                    ${{r.authors.join(', ')}} • ${{r.published_date || 'N/A'}}
                                </div>
                            </div>
                            <div class="match-badge ${{badgeClass}}">${{scorePct}}% Match</div>
                        </div>
                        
                        ${{r.has_text_snippet 
                            ? `<div class="snippet-box">"...${{r.snippet}}..."</div>`
                            : `<div class="warning">⚠️ No text snippet available (Copyright Restricted)</div>`
                        }}
                        
                        <div style="display:flex; gap:15px; font-size:13px; margin-top:10px;">
                            ${{r.url ? `<a href="${{r.url}}" target="_blank">View on Google Books</a>` : ''}}
                            <span class="raw-toggle" onclick="this.nextElementSibling.style.display='block'">Show Raw Debug Data</span>
                            <div class="raw-data">${{JSON.stringify(r, null, 2)}}</div>
                        </div>
                    </div>`;
                }});
                return html;
            }}

            // --- TEST HELPERS ---
            function loadTest(scenario) {{
                const inputs = {{
                    'prozac': {{ q: "By the time Tess's story had played itself out, I had seen perhaps a dozen", a: "Kramer" }},
                    'einstein': {{ q: "Imagination is more important than knowledge.", a: "Einstein" }},
                    'potter': {{ q: "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say", a: "Rowling" }}
                }};
                document.getElementById('quote').value = inputs[scenario].q;
                document.getElementById('author').value = inputs[scenario].a;
            }}
        </script>
    </body>
    </html>
    """

@app.post("/api/search")
async def google_api_search(request: SearchRequest) -> List[Dict[str, Any]]:
    results = await search_google_books_api(
        request.quote, request.author_hint, request.max_results, request.use_exact_phrase
    )
    return [asdict(r) for r in results]

@app.post("/serpapi/search")
async def serpapi_search(request: SearchRequest) -> List[Dict[str, Any]]:
    results = await search_serpapi_books(
        request.quote, request.author_hint, request.max_results
    )
    return [asdict(r) for r in results]

@app.get("/config")
async def config():
    """Secure configuration check."""
    return {
        "google_ok": bool(GOOGLE_BOOKS_API_KEY),
        "serp_ok": bool(SERPAPI_KEY)
    }

# =============================================================================
# STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
