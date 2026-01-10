#!/usr/bin/env python3
"""
Google Books Standalone Test App (v2.3 - Extended Text Fetch)
==============================================================
REVISIONS v2.3:
- Added volume details fetch to try getting extended text
- Logs full API response structure for debugging
- Checks description, layerInfo for additional text content
REVISIONS v2.2:
- Added whitespace normalization before comparison (fixes alignment issues)
- Google Books snippets have extra spaces around punctuation (" ; " vs "; ")
- This threw off SequenceMatcher alignment, hiding single-char errors like "preparin" → "preparing"
REVISIONS v2.1:
- Filter empty/whitespace-only diffs (eliminates spurious "" diff noise)
- Filter boundary deletions at position 0 and end (snippet truncation artifacts)
- Added stripHtml() for raw snippet display (removes Google Books <b> tags)
REVISIONS v2.0:
- Added Phase 0 anchor window search (typo-resistant case identification)
- Added best_results tracking across all phases (returns with warning if no threshold met)
- Added source_quote field for side-by-side comparison UI
- Fixed autojunk=False in SequenceMatcher (critical bug fix)
- Added /health endpoint
- compute_match_with_diffs now returns 4 values (score, diffs, verified_html, source_quote)
REVISIONS v1.6:
- Split at em-dashes before extracting distinctive window
- Fixes: "basin—Seymour" no longer blocks finding "buspirone"
REVISIONS v1.5:
- Dynamic match threshold adjusts for quote/snippet length mismatch
REVISIONS v1.4:
- Search anchored to most distinctive word (drug names, citations, etc.)
- 200-char window extracted from distinctive word position
REVISIONS v1.3:
- NFC Unicode normalization (preserves §, ¶, accented chars)
- Fuzzy matching phase with 90% threshold (handles typos/OCR errors)
- Keyword fallback with 50% threshold
"""

import os
import re
import html
import logging
import unicodedata
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
    source_quote: str = ""  # Authentic text from source for side-by-side

@dataclass
class SearchResponse:
    results: List[BookMatch]
    trace: List[str] = field(default_factory=list)

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="Google Books Test App v2.3")

class SearchRequest(BaseModel):
    quote: str
    author_hint: Optional[str] = None

# =============================================================================
# LOGIC & HELPERS
# =============================================================================

# Pattern for dashes that should split text (like ellipsis does)
# Em-dash often joins clauses without spaces: "basin—Seymour"
# Also catches hyphen since browsers/forms often convert em-dash to hyphen
DASH_SPLIT_PATTERN = re.compile(r'[-—–―]')  # Hyphen, em-dash, en-dash, horizontal bar


def clean_quote_text(text: str) -> str:
    """Clean special characters for API acceptance while preserving Unicode symbols."""
    text = text.strip().strip('"').strip('\u201C').strip('\u201D')
    text = text.replace('\u2019', "'").replace('\u2018', "'")  # Curly apostrophes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Curly quotes
    # NOTE: We no longer convert em-dash to hyphen here - we split at it instead
    # Remove double quotes (we wrap in our own for exact phrase search)
    text = text.replace('"', '')
    # Normalize Unicode to NFC (preserves §, ¶, accented chars)
    text = unicodedata.normalize('NFC', text)
    return ' '.join(text.split())  # Normalize whitespace


def split_at_dashes(text: str) -> str:
    """
    Split text at em-dashes and return the segment with the most distinctive word.
    
    Em-dashes often join clauses without spaces ("basin—Seymour"), which creates
    tokens that won't match the source text. Like ellipsis handling, we split
    and take the best segment.
    
    Example:
        Input:  "One for the basin—Seymour stops taking the buspirone."
        Output: "Seymour stops taking the buspirone."  (has buspirone, score 100)
    
    NOTE: This function calls score_word_distinctiveness, so it must be defined
    after that function. We use a forward reference pattern here.
    """
    # Defer to implementation after score_word_distinctiveness is defined
    return _split_at_dashes_impl(text)


# Stop words for keyword extraction fallback
STOP_WORDS = {
    'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'by', 'at', 'and', 'or',
    'is', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'with',
    'as', 'from', 'are', 'not', 'but', 'if', 'then', 'than', 'so', 'no',
    'yes', 'all', 'any', 'each', 'which', 'who', 'whom', 'what', 'when',
    'where', 'why', 'how', 'out', 'said', 'says', 'told', 'asked', 'replied',
    'thought', 'knew', 'saw', 'came', 'went', 'made', 'took', 'gave', 'got',
    'man', 'men', 'woman', 'women', 'people', 'thing', 'things', 'time', 'way',
}


def extract_keywords_for_search(text: str, max_keywords: int = 10) -> List[str]:
    """Extract distinctive keywords from quote text for fallback search."""
    keywords = []
    seen = set()
    
    def add_keyword(word: str):
        word_lower = word.lower().strip()
        if word_lower and word_lower not in seen and len(word_lower) > 2 and word_lower not in STOP_WORDS:
            keywords.append(word_lower)
            seen.add(word_lower)
    
    # Priority 1: Extract 4-digit years
    years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text)
    for year in years:
        add_keyword(year)
    
    # Priority 2: Extract remaining distinctive words
    text_clean = re.sub(r'[^\w\s]', ' ', text[:300].lower())
    for word in text_clean.split():
        add_keyword(word)
    
    return keywords[:max_keywords]


def score_word_distinctiveness(word: str) -> int:
    """
    Score a word's distinctiveness for search anchor selection.
    Higher score = more distinctive = better search anchor.
    
    Scoring hierarchy:
    - Drug/chemical names: 100
    - Legal citations (§): 90
    - Numbers (statute refs, years): 80
    - Long words (10+ chars): 70
    - Medium words (7-9 chars): 50
    - Proper nouns (capitalized): 40
    - Short words not in stop list: 20
    - Stop words: 0
    """
    word_lower = word.lower().strip()
    word_clean = re.sub(r'[^\w]', '', word_lower)
    
    if not word_clean or len(word_clean) < 2:
        return 0
    
    if word_lower in STOP_WORDS:
        return 0
    
    # Drug/chemical patterns (ends in -ine, -one, -ole, -ate, -ide, etc.)
    drug_suffixes = ('pirone', 'prine', 'zepam', 'olan', 'etine', 'amine', 
                     'azole', 'mycin', 'cillin', 'statin', 'pril', 'sartan',
                     'olol', 'dipine', 'oxacin', 'cycline', 'dronate')
    if any(word_clean.endswith(suffix) for suffix in drug_suffixes):
        return 100
    
    # Legal citation symbols
    if '§' in word or word_lower in ('u.s.', 'm.r.s.', 'f.2d', 'f.3d', 'f.supp'):
        return 90
    
    # Numbers (statute references, years, etc.)
    if re.match(r'^\d+$', word_clean):
        if len(word_clean) == 4:  # Year
            return 85
        return 80
    
    # Long words are usually more distinctive
    if len(word_clean) >= 10:
        return 70
    
    if len(word_clean) >= 7:
        return 50
    
    # Capitalized words (proper nouns) - check original word
    if word and word[0].isupper() and len(word_clean) >= 3:
        return 40
    
    # Everything else not in stop words
    if len(word_clean) >= 3:
        return 20
    
    return 0


def _split_at_dashes_impl(text: str) -> str:
    """
    Implementation of split_at_dashes (called after score_word_distinctiveness is defined).
    """
    if not DASH_SPLIT_PATTERN.search(text):
        return text
    
    segments = DASH_SPLIT_PATTERN.split(text)
    segments = [s.strip() for s in segments if s.strip()]
    
    if not segments:
        return text
    
    if len(segments) == 1:
        return segments[0]
    
    # Find segment with highest distinctiveness score
    best_segment = segments[0]
    best_score = -1
    
    for seg in segments:
        # Score each word in segment, take max
        for match in re.finditer(r'\S+', seg):
            word = match.group()
            score = score_word_distinctiveness(word)
            if score > best_score:
                best_score = score
                best_segment = seg
    
    logger.info(f"Split at dash: {len(segments)} segments, best score={best_score}, selected: '{best_segment[:50]}...'")
    return best_segment


def extract_distinctive_window(text: str, max_chars: int = 200) -> str:
    """
    Extract a search window starting from the most distinctive word.
    
    Returns up to max_chars starting from the highest-scoring word's position.
    """
    if len(text) <= max_chars:
        return text
    
    # Tokenize while preserving positions
    words_with_pos = []
    for match in re.finditer(r'\S+', text):
        words_with_pos.append((match.group(), match.start(), match.end()))
    
    if not words_with_pos:
        return text[:max_chars]
    
    # Score each word
    best_score = -1
    best_pos = 0
    
    for word, start, end in words_with_pos:
        score = score_word_distinctiveness(word)
        if score > best_score:
            best_score = score
            best_pos = start
    
    # Extract window starting at best position
    window = text[best_pos:best_pos + max_chars]
    
    logger.info(f"Distinctive window: score={best_score}, starts at char {best_pos}: '{window[:50]}...'")
    
    return window


def extract_anchor_window(text: str, window_size: int = 40, min_score: int = 50) -> Optional[str]:
    """
    Extract a character window around the most distinctive word in text.
    
    Layer 1 of two-layer search: Creates a search string anchored to a
    distinctive word, avoiding typo-prone common words.
    
    Window positioning based on anchor location:
    - First 25% of text:  anchor + chars after
    - Middle 50% of text: chars before + anchor + chars after (centered)
    - Last 25% of text:   chars before + anchor
    
    Args:
        text: User's quote text
        window_size: Total characters to extract around anchor (default 40)
        min_score: Minimum distinctiveness score to qualify (default 50)
        
    Returns:
        Character window string, or None if no suitable anchor found
    """
    if not text or len(text) < 20:
        return None
    
    # Find all words with positions
    best_anchor = None
    best_score = -1
    best_start = 0
    best_end = 0
    
    for match in re.finditer(r'§\s*\d+|\d+\s*[A-Z]\.[A-Z]\.[A-Z]\.|\S+', text):
        word = match.group()
        score = score_word_distinctiveness(word)
        if score > best_score and score >= min_score:
            best_score = score
            best_anchor = word
            best_start = match.start()
            best_end = match.end()
    
    if not best_anchor:
        logger.info("No suitable anchor found (all words below min_score)")
        return None
    
    text_len = len(text)
    anchor_pos_ratio = best_start / text_len
    
    # Determine window boundaries based on anchor position
    if anchor_pos_ratio < 0.25:
        # Anchor near start: take anchor + chars after
        win_start = best_start
        win_end = min(text_len, best_end + window_size)
    elif anchor_pos_ratio > 0.75:
        # Anchor near end: take chars before + anchor
        win_start = max(0, best_start - window_size)
        win_end = best_end
    else:
        # Anchor in middle: center the window
        half_window = window_size // 2
        win_start = max(0, best_start - half_window)
        win_end = min(text_len, best_end + half_window)
    
    window = text[win_start:win_end].strip()
    
    logger.info(f"Anchor window: '{best_anchor}' (score={best_score}) at {anchor_pos_ratio:.0%} → '{window[:50]}...'")
    
    return window


def compute_dynamic_threshold(quote_len: int, snippet_len: int, base_threshold: float = 0.90) -> float:
    """
    Adjust match threshold based on length ratio.
    
    SequenceMatcher.ratio() = 2 * matches / (len_a + len_b)
    When snippet is shorter than quote, perfect overlap is capped.
    
    Example: quote=350, snippet=200, perfect overlap=200 chars
             max_ratio = 2*200 / (350+200) = 0.727
             
    We require base_threshold (90%) of what's theoretically achievable.
    """
    if snippet_len >= quote_len:
        return base_threshold
    
    # Max possible ratio for perfect overlap of shorter string
    max_possible = (2 * snippet_len) / (quote_len + snippet_len)
    
    # Require 90% of what's theoretically achievable
    adjusted = base_threshold * max_possible
    
    # Floor at 40% to avoid false positives
    return max(adjusted, 0.40)


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
    
    # CRITICAL: autojunk=False prevents SequenceMatcher from ignoring common chars
    return SequenceMatcher(None, u, s, autojunk=False).ratio()


def compute_match_with_diffs(user_quote: str, source_text: str) -> tuple:
    """
    Computes match score AND identifies character-level differences.
    Returns: (score, diffs_list, verified_quote_html, source_quote)
    
    Filters out:
    - Empty/whitespace-only diffs (noise from SequenceMatcher)
    - Boundary deletions at position 0 (snippet truncation artifacts)
    """
    if not user_quote or not source_text:
        return 0.0, [], user_quote, ""
    
    # Strip HTML tags and decode entities from snippet
    clean_source = re.sub(r'<[^>]+>', '', source_text)
    clean_source = html.unescape(clean_source)
    
    # Light normalization for matching (preserve case for display)
    def normalize_for_match(t):
        t = t.replace('\u2019', "'").replace('\u2018', "'")
        t = t.replace('\u201c', '"').replace('\u201d', '"')
        t = t.replace('\u2014', '-').replace('\u2013', '-')
        return t
    
    # Normalize whitespace for COMPARISON only (not display)
    # Google Books snippets have extra spaces around punctuation (" ; " instead of "; ")
    # which throws off SequenceMatcher alignment
    def normalize_whitespace(t):
        # Remove space BEFORE punctuation ("; " stays, " ; " becomes "; ")
        t = re.sub(r'\s+([;:,.\?!])', r'\1', t)
        # Collapse multiple spaces to single space
        t = ' '.join(t.split())
        return t
    
    user_norm = normalize_for_match(user_quote)
    source_norm = normalize_for_match(clean_source)
    
    # For comparison: also normalize whitespace
    user_compare = normalize_whitespace(user_norm.lower())
    source_compare = normalize_whitespace(source_norm.lower())
    
    # Use SequenceMatcher to find differences
    # CRITICAL: autojunk=False prevents SequenceMatcher from ignoring common chars
    matcher = SequenceMatcher(None, user_compare, source_compare, autojunk=False)
    score = matcher.ratio()
    
    diffs = []
    verified_html = ""
    user_len = len(user_compare)
    
    # Get matching blocks and identify differences
    opcodes = matcher.get_opcodes()
    
    # Map comparison positions back to display positions
    # (they differ because normalize_whitespace changes lengths)
    # Strategy: Track position in user_norm (display) separately
    display_pos = 0
    
    for tag, i1, i2, j1, j2 in opcodes:
        # Get segments from COMPARISON strings (for diff logic)
        user_segment_compare = user_compare[i1:i2]
        source_segment_compare = source_compare[j1:j2]
        
        # Calculate how many chars to advance in display string
        # This is approximate but works for most cases
        segment_len = i2 - i1
        user_segment_display = user_norm[display_pos:display_pos + segment_len] if segment_len > 0 else ""
        
        # For source segment in diffs, use the comparison version (it's cleaner)
        source_segment = source_segment_compare
        
        if tag == 'equal':
            # Matching text - no highlight
            verified_html += html.escape(user_segment_display)
            display_pos += segment_len
        elif tag == 'replace':
            # Skip empty/whitespace-only diffs (noise)
            if not user_segment_compare.strip() and not source_segment_compare.strip():
                verified_html += html.escape(user_segment_display)
                display_pos += segment_len
                continue
            # Substitution - user wrote something different
            diffs.append(DiffSegment(
                position=i1,
                user_text=user_segment_compare,
                source_text=source_segment,
                diff_type='substitution'
            ))
            verified_html += f'<span class="diff-error" title="Source: {html.escape(source_segment)}">{html.escape(user_segment_display)}</span>'
            display_pos += segment_len
        elif tag == 'insert':
            # SequenceMatcher 'insert' means: SOURCE has text that USER is missing
            # (need to insert source text into user to make them match)
            # Skip empty/whitespace-only diffs (noise)
            if not source_segment_compare.strip():
                continue
            # Skip boundary deletions (snippet truncation artifacts)
            is_start_boundary = (i1 == 0 and j1 == 0)
            is_end_boundary = (i2 >= user_len - 5)
            if is_start_boundary or is_end_boundary:
                continue
            # User is MISSING text that's in source (deletion error)
            diffs.append(DiffSegment(
                position=i1,
                user_text="",
                source_text=source_segment,
                diff_type='deletion'
            ))
            verified_html += f'<span class="diff-missing" title="Missing: {html.escape(source_segment)}">[...]</span>'
        elif tag == 'delete':
            # SequenceMatcher 'delete' means: USER has text that SOURCE doesn't have
            # (need to delete user text to make it match source)
            # Skip empty/whitespace-only diffs (noise)
            if not user_segment_compare.strip():
                verified_html += html.escape(user_segment_display)
                display_pos += segment_len
                continue
            # User ADDED text not in source (insertion error)
            diffs.append(DiffSegment(
                position=i1,
                user_text=user_segment_compare,
                source_text="",
                diff_type='insertion'
            ))
            verified_html += f'<span class="diff-error" title="Not in source">{html.escape(user_segment_display)}</span>'
            display_pos += segment_len
    
    return score, diffs, verified_html, source_norm

# =============================================================================
# CASCADING SEARCH LOGIC
# =============================================================================

async def search_google_books_cascading(quote: str, author: str) -> SearchResponse:
    """
    Tries multiple search strategies until one works:
    0. Anchor window search (typo-resistant case identification)
    1. Distinctive window (200 chars from most distinctive word)
    2. Fuzzy matching (90% threshold)
    3. Keyword fallback (50% threshold)
    
    Returns best available results with warnings if no strategy meets threshold.
    """
    trace = []
    clean_q = clean_quote_text(quote)
    
    # Split at em-dashes and take segment with most distinctive word
    clean_q = split_at_dashes(clean_q)
    
    # Extract 40-char window around best anchor for Phase 0 (typo-resistant)
    anchor_window = extract_anchor_window(clean_q, window_size=40, min_score=50)
    
    # Extract 200-char window starting from most distinctive word
    distinctive_q = extract_distinctive_window(clean_q, max_chars=200)
    
    # Also prepare shorter fragment for fallback
    words = distinctive_q.split()
    short_q = " ".join(words[:15]) if len(words) > 15 else distinctive_q
    
    trace.append(f"Search window: {distinctive_q[:60]}...")
    
    # Track best results found across all phases (for showing with errors)
    best_results = []
    
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        
        # =================================================================
        # PHASE 0: Anchor window search (typo-resistant identification)
        # =================================================================
        if anchor_window:
            trace.append(f"Phase 0 - Anchor window: {anchor_window[:50]}...")
            logger.info(f"Phase 0 - Trying anchor window: {anchor_window}")
            
            search_query = anchor_window  # No quotes for fuzzy matching
            if author:
                search_query += f" inauthor:{author}"
            
            try:
                resp = await client.get(GOOGLE_BOOKS_BASE_URL, params={
                    "q": search_query, "key": GOOGLE_BOOKS_API_KEY, "maxResults": 10, "printType": "books"
                })
                data = resp.json()
                items = data.get("items", [])
                
                if items:
                    trace.append(f"Phase 0 - Found {len(items)} candidate(s), verifying...")
                    logger.info(f"Phase 0 - Found {len(items)} candidates")
                    
                    parsed = await parse_google_items_async(client, items, quote)
                    # Sort by match score descending
                    parsed.sort(key=lambda r: r.match_score, reverse=True)
                    
                    # Check for high-confidence matches (>= 80%)
                    for r in parsed:
                        if r.match_score >= 0.80:
                            trace.append(f"✅ Phase 0 verified: {r.match_score:.0%} match in '{r.title[:30]}...'")
                            logger.info(f"✅ Phase 0 success: {r.match_score:.0%} match")
                            return SearchResponse(results=[r], trace=trace)
                    
                    # Track best for potential fallback return
                    if parsed and (not best_results or parsed[0].match_score > best_results[0].match_score):
                        best_results = parsed[:5]
                    
                    trace.append(f"Phase 0 - No candidate >= 80% (best: {parsed[0].match_score:.0%})")
                else:
                    trace.append("Phase 0 - No candidates found")
                    logger.info("Phase 0 - 0 results from anchor window search")
            except Exception as e:
                trace.append(f"⚠️ Phase 0 Error: {str(e)}")
        
        # =================================================================
        # PHASE 1: Exact phrase strategies (trusted, match_score = 1.0)
        # =================================================================
        exact_strategies = [
            {"name": "Distinctive Window + Author", "q": distinctive_q, "auth": author},
            {"name": "Distinctive Window Only", "q": distinctive_q, "auth": None},
            {"name": "Short Fragment + Author", "q": short_q, "auth": author},
            {"name": "Short Fragment Only", "q": short_q, "auth": None},
        ]
        
        for strategy in exact_strategies:
            # Skip author strategies if no author provided
            if strategy["auth"] is not None and not author:
                continue

            query = f'"{strategy["q"]}"'  # Exact phrase matching
            if strategy["auth"]:
                query += f" inauthor:{strategy['auth']}"
            
            trace.append(f"Trying: {strategy['name']}...")
            
            try:
                resp = await client.get(GOOGLE_BOOKS_BASE_URL, params={
                    "q": query, "key": GOOGLE_BOOKS_API_KEY, "maxResults": 5, "printType": "books"
                })
                data = resp.json()
                items = data.get("items", [])
                
                if items:
                    parsed = await parse_google_items_async(client, items, quote)
                    # EXACT PHRASE SEARCH: Trust the match
                    for r in parsed:
                        r.match_score = 1.0
                    
                    trace.append(f"✅ Found {len(parsed)} result(s) via exact phrase match")
                    return SearchResponse(results=parsed, trace=trace)
                else:
                    trace.append("❌ 0 results.")
            except Exception as e:
                trace.append(f"⚠️ Error: {str(e)}")
        
        # =================================================================
        # PHASE 2: Fuzzy strategies (no quotes, verify with 90% threshold)
        # =================================================================
        trace.append("Exact phrase exhausted, trying fuzzy matching...")
        
        fuzzy_strategies = [
            {"name": "Fuzzy Distinctive Window", "q": distinctive_q},
            {"name": "Fuzzy Short Fragment", "q": short_q},
        ]
        
        for strategy in fuzzy_strategies:
            search_query = strategy["q"]  # NO quotes = fuzzy matching
            if author:
                search_query += f" inauthor:{author}"
            
            trace.append(f"Trying: {strategy['name']}...")
            
            try:
                resp = await client.get(GOOGLE_BOOKS_BASE_URL, params={
                    "q": search_query, "key": GOOGLE_BOOKS_API_KEY, "maxResults": 10, "printType": "books"
                })
                data = resp.json()
                items = data.get("items", [])
                
                if items:
                    parsed = await parse_google_items_async(client, items, quote)
                    # Sort by match score descending
                    parsed.sort(key=lambda r: r.match_score, reverse=True)
                    
                    # Filter by DYNAMIC threshold (adjusts for length mismatch)
                    quote_len = len(quote)
                    verified = []
                    for r in parsed:
                        snippet_len = len(r.snippet) if r.snippet else 0
                        threshold = compute_dynamic_threshold(quote_len, snippet_len)
                        if r.match_score >= threshold:
                            verified.append(r)
                            trace.append(f"   ↳ '{r.title[:30]}...' score={r.match_score:.2f} >= threshold={threshold:.2f}")
                    
                    if verified:
                        trace.append(f"✅ Found {len(verified)} result(s) above dynamic threshold")
                        return SearchResponse(results=verified, trace=trace)
                    else:
                        # Track best for potential fallback return
                        if parsed and (not best_results or parsed[0].match_score > best_results[0].match_score):
                            best_results = parsed[:5]
                        trace.append(f"❌ {len(parsed)} results but none above dynamic threshold (best: {parsed[0].match_score:.0%})")
                else:
                    trace.append("❌ 0 results.")
            except Exception as e:
                trace.append(f"⚠️ Error: {str(e)}")
        
        # =================================================================
        # PHASE 3: Keyword fallback (dynamic threshold, base 50%)
        # =================================================================
        trace.append("Fuzzy matching exhausted, trying keyword fallback...")
        keywords = extract_keywords_for_search(quote, max_keywords=10)
        
        if keywords:
            keyword_query = ' '.join(keywords)
            if author:
                keyword_query += f" inauthor:{author}"
            
            trace.append(f"Keywords: {keyword_query}")
            
            try:
                resp = await client.get(GOOGLE_BOOKS_BASE_URL, params={
                    "q": keyword_query, "key": GOOGLE_BOOKS_API_KEY, "maxResults": 10, "printType": "books"
                })
                data = resp.json()
                items = data.get("items", [])
                
                if items:
                    parsed = await parse_google_items_async(client, items, quote)
                    # Sort by match score descending
                    parsed.sort(key=lambda r: r.match_score, reverse=True)
                    
                    # Dynamic threshold with lower base for keyword fallback
                    quote_len = len(quote)
                    verified = []
                    for r in parsed:
                        snippet_len = len(r.snippet) if r.snippet else 0
                        threshold = compute_dynamic_threshold(quote_len, snippet_len, base_threshold=0.50)
                        if r.match_score >= threshold:
                            verified.append(r)
                    
                    if verified:
                        trace.append(f"✅ Found {len(verified)} result(s) via keyword fallback")
                        return SearchResponse(results=verified, trace=trace)
                    else:
                        # Track best for potential fallback return
                        if parsed and (not best_results or parsed[0].match_score > best_results[0].match_score):
                            best_results = parsed[:5]
                        trace.append(f"❌ Keyword results below dynamic threshold (best: {parsed[0].match_score:.0%})")
                else:
                    trace.append("❌ 0 keyword results.")
            except Exception as e:
                trace.append(f"⚠️ Error: {str(e)}")
        
        # =================================================================
        # Return best available results (with warning)
        # =================================================================
        if best_results:
            trace.append(f"⚠️ Returning best available matches (may contain errors)")
            logger.info(f"⚠️ Returning {len(best_results)} best available results with potential errors")
            return SearchResponse(results=best_results, trace=trace)

    trace.append("⛔ All strategies exhausted - no results found")
    return SearchResponse(results=[], trace=trace)

async def fetch_volume_details(client: httpx.AsyncClient, volume_id: str) -> dict:
    """
    Fetch detailed volume info which may include more text content.
    """
    try:
        url = f"https://www.googleapis.com/books/v1/volumes/{volume_id}"
        resp = await client.get(url, params={"key": GOOGLE_BOOKS_API_KEY})
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch volume details for {volume_id}: {e}")
    return {}


async def parse_google_items_async(client: httpx.AsyncClient, items, original_quote):
    """
    Parse Google Books items and try to fetch extended text from volume details.
    """
    parsed = []
    for item in items:
        vol = item.get("volumeInfo", {})
        volume_id = item.get("id", "")
        snip = item.get("searchInfo", {}).get("textSnippet", "")
        
        # Log full item structure for first result (debug)
        if not parsed:
            logger.info(f"First result keys: {list(item.keys())}")
            if "accessInfo" in item:
                logger.info(f"accessInfo keys: {list(item.get('accessInfo', {}).keys())}")
        
        # Try to get extended text from volume details
        extended_text = snip
        if volume_id:
            details = await fetch_volume_details(client, volume_id)
            if details:
                # Check various fields that might have more text
                search_info = details.get("searchInfo", {})
                if search_info.get("textSnippet"):
                    detail_snip = search_info.get("textSnippet", "")
                    if len(detail_snip) > len(snip):
                        extended_text = detail_snip
                        logger.info(f"Extended snippet from volume details: {len(snip)} → {len(detail_snip)} chars")
                
                # Check for layerInfo or other text fields
                layer_info = details.get("layerInfo", {})
                if layer_info:
                    logger.info(f"layerInfo keys: {list(layer_info.keys())}")
                
                # Check volumeInfo for description (sometimes contains quotes)
                vol_info = details.get("volumeInfo", {})
                description = vol_info.get("description", "")
                if description and original_quote[:50].lower() in description.lower():
                    logger.info(f"Quote found in description! Length: {len(description)}")
                    if len(description) > len(extended_text):
                        extended_text = description
        
        # Compute score and detect differences
        score, diffs, verified_html, source_quote = compute_match_with_diffs(original_quote, extended_text)
        
        parsed.append(BookMatch(
            title=vol.get("title", "Unknown"),
            authors=vol.get("authors", []),
            match_score=score,
            snippet=extended_text,  # Use extended text
            has_text_snippet=bool(extended_text),
            url=vol.get("previewLink", ""),
            published_date=vol.get("publishedDate", ""),
            source="google_api",
            diffs=[asdict(d) for d in diffs],
            verified_quote=verified_html,
            source_quote=source_quote
        ))
    return parsed


def parse_google_items(items, original_quote):
    parsed = []
    for item in items:
        vol = item.get("volumeInfo", {})
        snip = item.get("searchInfo", {}).get("textSnippet", "")
        
        # Compute score and detect differences (now returns 4 values)
        score, diffs, verified_html, source_quote = compute_match_with_diffs(original_quote, snip)
        
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
            verified_quote=verified_html,
            source_quote=source_quote
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

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "google_api_configured": bool(GOOGLE_BOOKS_API_KEY),
        "serpapi_configured": bool(SERPAPI_KEY),
        "google_api_key_length": len(GOOGLE_BOOKS_API_KEY)
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Google Books Tester v2.3</title>
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
            .comparison-container { display: flex; gap: 15px; margin: 15px 0; }
            .comparison-panel { flex: 1; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
            .comparison-header { padding: 10px 15px; font-weight: bold; font-size: 0.9em; }
            .comparison-header.user { background: #e3f2fd; color: #1565c0; }
            .comparison-header.source { background: #e8f5e9; color: #2e7d32; }
            .comparison-body { padding: 15px; background: #fff; line-height: 1.6; min-height: 100px; }
            .comparison-body.user { border-top: 3px solid #1565c0; }
            .comparison-body.source { border-top: 3px solid #2e7d32; }
        </style>
    </head>
    <body>
        <h1>📚 Google Books Tester v2.3</h1>
        <p style="color:#666">Now with side-by-side quotation accuracy verification</p>
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
            
            function escapeHtml(text) {
                if (!text) return '';
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
            
            function stripHtml(text) {
                if (!text) return '';
                const div = document.createElement('div');
                div.innerHTML = text;
                return div.textContent || div.innerText || '';
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
                            <div>Match Score: ${Math.round(r.match_score*100)}%</div>`;
                        
                        // Side-by-side comparison
                        if (r.verified_quote && r.source_quote) {
                            html += `<div class="comparison-container">
                                <div class="comparison-panel">
                                    <div class="comparison-header user">📝 Your Quotation</div>
                                    <div class="comparison-body user">${r.verified_quote}</div>
                                </div>
                                <div class="comparison-panel">
                                    <div class="comparison-header source">✓ Authentic Source</div>
                                    <div class="comparison-body source">${escapeHtml(r.source_quote)}</div>
                                </div>
                            </div>`;
                        } else if (r.verified_quote) {
                            html += `<div style="margin-top:10px"><strong>Your Quotation (${statusText}):</strong></div>
                                <div class="verified-quote">${r.verified_quote}</div>`;
                        }
                        
                        if(hasErrors) {
                            html += `<div class="error-summary">
                                <strong>📋 Detected Differences (${r.diffs.length}):</strong>`;
                            r.diffs.forEach((d, i) => {
                                let desc = '';
                                if(d.diff_type === 'substitution') {
                                    desc = `You wrote "<b>${escapeHtml(d.user_text)}</b>" → Source has "<b>${escapeHtml(d.source_text)}</b>"`;
                                } else if(d.diff_type === 'insertion') {
                                    desc = `"<b>${escapeHtml(d.user_text)}</b>" not found in source`;
                                } else if(d.diff_type === 'deletion') {
                                    desc = `Missing from your quote: "<b>${escapeHtml(d.source_text)}</b>"`;
                                }
                                html += `<div class="error-item">${i+1}. ${desc}</div>`;
                            });
                            html += `</div>`;
                        }
                        
                        html += `<div class="snippet-label">Source snippet from Google Books:</div>
                            <div class="snippet-text">"${stripHtml(r.snippet)}"</div>
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
