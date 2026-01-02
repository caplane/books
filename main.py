#!/usr/bin/env python3
"""
Google Books Standalone Test App
================================

Isolated test environment for debugging Google Books API integration.
Tests BOTH the native Google Books API AND SerpAPI Google Books.

Endpoints:
    GET  /              - Web UI for testing
    POST /api/search    - Native Google Books API search
    POST /serpapi/search - SerpAPI Google Books search
    GET  /health        - Health check
    GET  /config        - Show configuration status
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
    version="1.0.0"
)

# =============================================================================
# REQUEST MODELS
# =============================================================================

class SearchRequest(BaseModel):
    quote: str
    author_hint: Optional[str] = None
    max_results: int = 5
    use_exact_phrase: bool = True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_match_score(user_quote: str, source_text: str) -> float:
    """Compute similarity between user quote and source text."""
    user_clean = user_quote.lower().strip()
    source_clean = source_text.lower().strip()
    return SequenceMatcher(None, user_clean, source_clean).ratio()

# =============================================================================
# NATIVE GOOGLE BOOKS API
# =============================================================================

async def search_google_books_api(
    quote_text: str,
    author_hint: Optional[str] = None,
    max_results: int = 5,
    use_exact_phrase: bool = True
) -> List[BookMatch]:
    """Search using native Google Books API."""
    results = []
    
    logger.info("=" * 50)
    logger.info("GOOGLE BOOKS API SEARCH")
    logger.info("=" * 50)
    logger.info(f"Quote: '{quote_text[:60]}...'")
    logger.info(f"Author hint: {author_hint}")
    logger.info(f"API Key set: {bool(GOOGLE_BOOKS_API_KEY)}")
    logger.info(f"API Key length: {len(GOOGLE_BOOKS_API_KEY)}")
    
    if not GOOGLE_BOOKS_API_KEY:
        logger.error("No Google Books API key configured!")
        return [BookMatch(
            success=False, title="", authors=[], publisher="",
            published_date="", snippet="", page_number=None,
            url="", isbn="", match_score=0.0, source="google_api",
            error="GOOGLE_BOOKS_API_KEY not configured"
        )]
    
    # Build query
    search_text = quote_text[:60].strip()
    if use_exact_phrase:
        query = f'"{search_text}"'
    else:
        query = search_text
    
    if author_hint:
        query += f" inauthor:{author_hint}"
    
    params = {
        "q": query,
        "key": GOOGLE_BOOKS_API_KEY,
        "maxResults": max_results,
        "printType": "books"
    }
    
    logger.info(f"Query: {query}")
    logger.info(f"Full URL: {GOOGLE_BOOKS_BASE_URL}?q={query}&maxResults={max_results}")
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(GOOGLE_BOOKS_BASE_URL, params=params)
            
            logger.info(f"Response status: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            
            total_items = data.get("totalItems", 0)
            items = data.get("items", [])
            
            logger.info(f"totalItems: {total_items}")
            logger.info(f"items returned: {len(items)}")
            
            if total_items == 0:
                logger.warning("API returned 0 results - trying without quotes")
                # Retry without exact phrase
                if use_exact_phrase:
                    return await search_google_books_api(
                        quote_text, author_hint, max_results, use_exact_phrase=False
                    )
            
            for item in items[:max_results]:
                volume_info = item.get("volumeInfo", {})
                search_info = item.get("searchInfo", {})
                
                title = volume_info.get("title", "Unknown")
                authors = volume_info.get("authors", [])
                snippet = search_info.get("textSnippet", "")
                has_snippet = bool(snippet)
                
                logger.info(f"  Book: {title}")
                logger.info(f"    Authors: {authors}")
                logger.info(f"    Has textSnippet: {has_snippet}")
                
                if not snippet:
                    logger.debug(f"    Skipping - no textSnippet (preview restricted)")
                    # Still include it but note the issue
                
                # Compute match score
                match_score = compute_match_score(quote_text, snippet) if snippet else 0.0
                
                # Extract ISBN
                isbn = ""
                for identifier in volume_info.get("industryIdentifiers", []):
                    if identifier.get("type") == "ISBN_13":
                        isbn = identifier.get("identifier", "")
                        break
                    elif identifier.get("type") == "ISBN_10" and not isbn:
                        isbn = identifier.get("identifier", "")
                
                results.append(BookMatch(
                    success=True,
                    title=title,
                    authors=authors,
                    publisher=volume_info.get("publisher", ""),
                    published_date=volume_info.get("publishedDate", ""),
                    snippet=snippet,
                    page_number=None,
                    url=volume_info.get("previewLink", ""),
                    isbn=isbn,
                    match_score=match_score,
                    source="google_api",
                    has_text_snippet=has_snippet,
                    preview_link=volume_info.get("previewLink", ""),
                    info_link=volume_info.get("infoLink", "")
                ))
            
            # Sort by match score
            results.sort(key=lambda x: x.match_score, reverse=True)
            
            # Summary
            with_snippets = sum(1 for r in results if r.has_text_snippet)
            without_snippets = len(results) - with_snippets
            logger.info(f"Summary: {len(results)} books, {with_snippets} with snippets, {without_snippets} without")
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        results.append(BookMatch(
            success=False, title="", authors=[], publisher="",
            published_date="", snippet="", page_number=None,
            url="", isbn="", match_score=0.0, source="google_api",
            error=f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        ))
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
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
    
    logger.info("=" * 50)
    logger.info("SERPAPI GOOGLE BOOKS SEARCH")
    logger.info("=" * 50)
    logger.info(f"Quote: '{quote_text[:60]}...'")
    logger.info(f"Author hint: {author_hint}")
    logger.info(f"SERPAPI_KEY set: {bool(SERPAPI_KEY)}")
    logger.info(f"SERPAPI_KEY length: {len(SERPAPI_KEY)}")
    
    if not SERPAPI_KEY:
        logger.error("No SerpAPI key configured!")
        return [BookMatch(
            success=False, title="", authors=[], publisher="",
            published_date="", snippet="", page_number=None,
            url="", isbn="", match_score=0.0, source="serpapi",
            error="SERPAPI_KEY not configured"
        )]
    
    # Build query - use first 80 chars
    search_text = quote_text[:80].strip()
    if author_hint:
        search_text += f" {author_hint}"
    
    params = {
        "engine": "google_books",
        "q": search_text,
        "api_key": SERPAPI_KEY,
    }
    
    logger.info(f"Query: {search_text}")
    logger.info(f"Engine: google_books")
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            logger.info(f"Making request to: {SERPAPI_BASE_URL}")
            response = await client.get(SERPAPI_BASE_URL, params=params)
            
            logger.info(f"Response status: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            
            # Log all keys in response
            logger.info(f"Response keys: {list(data.keys())}")
            
            # Try different possible result keys
            books_results = data.get("books_results", [])
            organic_results = data.get("organic_results", [])
            inline_results = data.get("inline_results", [])
            
            logger.info(f"books_results count: {len(books_results)}")
            logger.info(f"organic_results count: {len(organic_results)}")
            logger.info(f"inline_results count: {len(inline_results)}")
            
            # Use books_results if available, otherwise try organic_results
            items = books_results or organic_results or inline_results
            
            logger.info(f"Using {len(items)} results")
            
            if not items:
                logger.warning("No results in any result field")
                # Log first 500 chars of response for debugging
                logger.info(f"Response preview: {str(data)[:500]}")
                return results
            
            for item in items[:max_results]:
                # Log the item structure
                logger.debug(f"Item keys: {list(item.keys())}")
                
                snippet = item.get("snippet", "") or item.get("description", "")
                if not snippet:
                    logger.debug(f"Skipping item - no snippet")
                    continue
                
                # Compute match score
                match_score = compute_match_score(quote_text, snippet)
                
                # Extract page from title (often "Book Title - Page 123")
                page_number = None
                title = item.get("title", "Unknown")
                if " - Page " in title:
                    parts = title.rsplit(" - Page ", 1)
                    title = parts[0]
                    try:
                        page_number = int(parts[1])
                    except (ValueError, IndexError):
                        pass
                
                # Parse authors
                authors_str = item.get("authors", "") or item.get("author", "")
                authors = [a.strip() for a in authors_str.split(",")] if authors_str else []
                
                logger.info(f"  Book: {title}")
                logger.info(f"    Authors: {authors}")
                logger.info(f"    Score: {match_score:.2f}")
                
                results.append(BookMatch(
                    success=True,
                    title=title,
                    authors=authors,
                    publisher=item.get("publisher", ""),
                    published_date=item.get("published_date", "") or item.get("date", ""),
                    snippet=snippet,
                    page_number=page_number,
                    url=item.get("link", "") or item.get("url", ""),
                    isbn="",
                    match_score=match_score,
                    source="serpapi",
                    has_text_snippet=bool(snippet)
                ))
            
            # Sort by match score
            results.sort(key=lambda x: x.match_score, reverse=True)
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        logger.error(f"Response: {e.response.text[:500]}")
        results.append(BookMatch(
            success=False, title="", authors=[], publisher="",
            published_date="", snippet="", page_number=None,
            url="", isbn="", match_score=0.0, source="serpapi",
            error=f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        ))
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}")
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
    google_status = "✅ Configured" if GOOGLE_BOOKS_API_KEY else "❌ Missing"
    serpapi_status = "✅ Configured" if SERPAPI_KEY else "❌ Missing"
    
    google_preview = f"{GOOGLE_BOOKS_API_KEY[:8]}...{GOOGLE_BOOKS_API_KEY[-4:]}" if GOOGLE_BOOKS_API_KEY else "Not set"
    serpapi_preview = f"{SERPAPI_KEY[:8]}...{SERPAPI_KEY[-4:]}" if SERPAPI_KEY else "Not set"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Google Books Test App</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 40px auto; padding: 20px; }}
            h1 {{ color: #333; }}
            .status {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            .status.ok {{ background: #d4edda; }}
            .status.error {{ background: #f8d7da; }}
            .status-row {{ display: flex; gap: 20px; }}
            .status-box {{ flex: 1; padding: 15px; border-radius: 8px; }}
            .status-box.ok {{ background: #d4edda; }}
            .status-box.error {{ background: #f8d7da; }}
            textarea {{ width: 100%; height: 80px; font-size: 14px; padding: 10px; }}
            input[type="text"] {{ width: 300px; padding: 10px; font-size: 14px; }}
            button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 10px 5px 10px 0; }}
            button:hover {{ background: #0056b3; }}
            button.serpapi {{ background: #28a745; }}
            button.serpapi:hover {{ background: #1e7e34; }}
            button.both {{ background: #6f42c1; }}
            button.both:hover {{ background: #5a32a3; }}
            .result {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #007bff; }}
            .result.serpapi {{ border-left-color: #28a745; }}
            .result.error {{ border-left-color: #dc3545; background: #fff5f5; }}
            .result h3 {{ margin: 0 0 10px 0; }}
            .meta {{ color: #666; font-size: 13px; margin: 3px 0; }}
            .score {{ background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-family: monospace; }}
            .score.high {{ background: #d4edda; color: #155724; }}
            .score.medium {{ background: #fff3cd; color: #856404; }}
            .score.low {{ background: #f8d7da; color: #721c24; }}
            pre {{ background: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 12px; }}
            #results {{ margin-top: 20px; }}
            .loading {{ color: #666; font-style: italic; }}
            .columns {{ display: flex; gap: 20px; }}
            .column {{ flex: 1; }}
            .warning {{ background: #fff3cd; color: #856404; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>📚 Google Books Test App</h1>
        
        <div class="status-row">
            <div class="status-box {'ok' if GOOGLE_BOOKS_API_KEY else 'error'}">
                <strong>Google Books API:</strong> {google_status}<br>
                <code>{google_preview}</code>
            </div>
            <div class="status-box {'ok' if SERPAPI_KEY else 'error'}">
                <strong>SerpAPI:</strong> {serpapi_status}<br>
                <code>{serpapi_preview}</code>
            </div>
        </div>
        
        <h2>Search for Quote</h2>
        <p>Enter a quote from a book:</p>
        <textarea id="quote" placeholder="By the time Tess's story had played itself out, I had seen perhaps a dozen"></textarea>
        <br>
        <label>Author (optional): <input type="text" id="author" placeholder="Kramer"></label>
        <br><br>
        
        <button onclick="searchGoogleAPI()">🔵 Google Books API</button>
        <button class="serpapi" onclick="searchSerpAPI()">🟢 SerpAPI Books</button>
        <button class="both" onclick="searchBoth()">🟣 Compare Both</button>
        
        <h2>Quick Tests</h2>
        <button onclick="testProzac()">Test: Listening to Prozac</button>
        <button onclick="testEinstein()">Test: Einstein Quote</button>
        <button onclick="testLoving()">Test: Loving v. Virginia</button>
        
        <div id="results"></div>
        
        <script>
            async function searchGoogleAPI() {{
                await doSearch('/api/search', 'Google Books API');
            }}
            
            async function searchSerpAPI() {{
                await doSearch('/serpapi/search', 'SerpAPI Books');
            }}
            
            async function searchBoth() {{
                const quote = document.getElementById('quote').value;
                const author = document.getElementById('author').value;
                if (!quote) return alert('Enter a quote');
                
                document.getElementById('results').innerHTML = '<p class="loading">Searching both APIs...</p>';
                
                try {{
                    const [googleRes, serpRes] = await Promise.all([
                        fetch('/api/search', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{quote, author_hint: author, max_results: 5}})
                        }}).then(r => r.json()),
                        fetch('/serpapi/search', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{quote, author_hint: author, max_results: 5}})
                        }}).then(r => r.json())
                    ]);
                    
                    displayComparison(googleRes, serpRes);
                }} catch (e) {{
                    document.getElementById('results').innerHTML = '<div class="result error"><h3>Error</h3><pre>' + e + '</pre></div>';
                }}
            }}
            
            async function doSearch(endpoint, name) {{
                const quote = document.getElementById('quote').value;
                const author = document.getElementById('author').value;
                if (!quote) return alert('Enter a quote');
                
                document.getElementById('results').innerHTML = '<p class="loading">Searching ' + name + '...</p>';
                
                try {{
                    const response = await fetch(endpoint, {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{quote, author_hint: author, max_results: 5}})
                    }});
                    const data = await response.json();
                    displayResults(data, name);
                }} catch (e) {{
                    document.getElementById('results').innerHTML = '<div class="result error"><h3>Error</h3><pre>' + e + '</pre></div>';
                }}
            }}
            
            function testProzac() {{
                document.getElementById('quote').value = "By the time Tess's story had played itself out, I had seen perhaps a dozen";
                document.getElementById('author').value = "Kramer";
            }}
            
            function testEinstein() {{
                document.getElementById('quote').value = "Imagination is more important than knowledge. Knowledge is limited.";
                document.getElementById('author').value = "Einstein";
            }}
            
            function testLoving() {{
                document.getElementById('quote').value = "There is patently no legitimate overriding purpose independent of invidious racial discrimination";
                document.getElementById('author').value = "";
            }}
            
            function scoreClass(score) {{
                if (score >= 0.7) return 'high';
                if (score >= 0.5) return 'medium';
                return 'low';
            }}
            
            function displayResults(results, source) {{
                let html = '<h3>' + source + ' Results (' + results.length + ')</h3>';
                
                for (const r of results) {{
                    const cssClass = r.source === 'serpapi' ? 'serpapi' : '';
                    if (r.error) {{
                        html += '<div class="result error ' + cssClass + '"><h3>Error</h3><p>' + r.error + '</p></div>';
                    }} else if (r.success) {{
                        html += '<div class="result ' + cssClass + '">';
                        html += '<h3>' + (r.title || 'Unknown') + '</h3>';
                        html += '<p class="meta"><strong>Authors:</strong> ' + (r.authors?.join(', ') || 'N/A') + '</p>';
                        html += '<p class="meta"><strong>Published:</strong> ' + (r.published_date || 'N/A') + '</p>';
                        html += '<p class="meta"><strong>Match Score:</strong> <span class="score ' + scoreClass(r.match_score) + '">' + (r.match_score * 100).toFixed(1) + '%</span></p>';
                        html += '<p class="meta"><strong>Has Snippet:</strong> ' + (r.has_text_snippet ? '✅' : '❌') + '</p>';
                        if (r.snippet) html += '<p><em>"' + r.snippet.substring(0, 200) + '..."</em></p>';
                        if (!r.has_text_snippet) html += '<p class="warning">⚠️ No text snippet - publisher may restrict API access</p>';
                        if (r.url) html += '<p><a href="' + r.url + '" target="_blank">View Book →</a></p>';
                        html += '</div>';
                    }}
                }}
                
                html += '<h3>Raw Response</h3><pre>' + JSON.stringify(results, null, 2) + '</pre>';
                document.getElementById('results').innerHTML = html;
            }}
            
            function displayComparison(googleResults, serpResults) {{
                let html = '<div class="columns">';
                
                // Google column
                html += '<div class="column">';
                html += '<h3>🔵 Google Books API (' + googleResults.length + ')</h3>';
                for (const r of googleResults) {{
                    if (r.error) {{
                        html += '<div class="result error"><p>' + r.error + '</p></div>';
                    }} else if (r.success) {{
                        html += '<div class="result">';
                        html += '<strong>' + (r.title || 'Unknown').substring(0, 40) + '</strong><br>';
                        html += '<span class="score ' + scoreClass(r.match_score) + '">' + (r.match_score * 100).toFixed(1) + '%</span>';
                        html += ' | Snippet: ' + (r.has_text_snippet ? '✅' : '❌');
                        html += '</div>';
                    }}
                }}
                html += '</div>';
                
                // SerpAPI column
                html += '<div class="column">';
                html += '<h3>🟢 SerpAPI Books (' + serpResults.length + ')</h3>';
                for (const r of serpResults) {{
                    if (r.error) {{
                        html += '<div class="result error serpapi"><p>' + r.error + '</p></div>';
                    }} else if (r.success) {{
                        html += '<div class="result serpapi">';
                        html += '<strong>' + (r.title || 'Unknown').substring(0, 40) + '</strong><br>';
                        html += '<span class="score ' + scoreClass(r.match_score) + '">' + (r.match_score * 100).toFixed(1) + '%</span>';
                        html += ' | Snippet: ' + (r.has_text_snippet ? '✅' : '❌');
                        html += '</div>';
                    }}
                }}
                html += '</div>';
                
                html += '</div>';
                
                html += '<h3>Raw Responses</h3>';
                html += '<pre>// Google Books API\\n' + JSON.stringify(googleResults, null, 2) + '</pre>';
                html += '<pre>// SerpAPI Books\\n' + JSON.stringify(serpResults, null, 2) + '</pre>';
                
                document.getElementById('results').innerHTML = html;
            }}
        </script>
    </body>
    </html>
    """

@app.post("/api/search")
async def google_api_search(request: SearchRequest) -> List[Dict[str, Any]]:
    """Search using native Google Books API."""
    results = await search_google_books_api(
        request.quote, 
        request.author_hint, 
        request.max_results,
        request.use_exact_phrase
    )
    return [asdict(r) for r in results]

@app.post("/serpapi/search")
async def serpapi_search(request: SearchRequest) -> List[Dict[str, Any]]:
    """Search using SerpAPI Google Books."""
    results = await search_serpapi_books(
        request.quote,
        request.author_hint,
        request.max_results
    )
    return [asdict(r) for r in results]

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "google_api_key_configured": bool(GOOGLE_BOOKS_API_KEY),
        "serpapi_key_configured": bool(SERPAPI_KEY)
    }

@app.get("/config")
async def config():
    """Show configuration status."""
    return {
        "GOOGLE_BOOKS_API_KEY": f"{GOOGLE_BOOKS_API_KEY[:8]}...{GOOGLE_BOOKS_API_KEY[-4:]}" if GOOGLE_BOOKS_API_KEY else "NOT SET",
        "SERPAPI_KEY": f"{SERPAPI_KEY[:8]}...{SERPAPI_KEY[-4:]}" if SERPAPI_KEY else "NOT SET",
        "env_vars_checked": {
            "GOOGLE_BOOKS_API_KEY": bool(os.getenv("GOOGLE_BOOKS_API_KEY")),
            "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
            "SERPAPI_KEY": bool(os.getenv("SERPAPI_KEY"))
        }
    }

# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("Google Books Test App Starting")
    logger.info("=" * 60)
    logger.info(f"Google Books API Key: {bool(GOOGLE_BOOKS_API_KEY)}")
    logger.info(f"SerpAPI Key: {bool(SERPAPI_KEY)}")
    if GOOGLE_BOOKS_API_KEY:
        logger.info(f"Google Key Preview: {GOOGLE_BOOKS_API_KEY[:8]}...")
    if SERPAPI_KEY:
        logger.info(f"SerpAPI Key Preview: {SERPAPI_KEY[:8]}...")
    logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
