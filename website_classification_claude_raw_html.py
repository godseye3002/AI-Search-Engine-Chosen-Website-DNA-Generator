import os
import json
import time
import re
from typing import Optional, List, Dict, Any, Set, Literal
from dataclasses import dataclass, field, asdict
from urllib.parse import urlparse, parse_qs, urljoin
from pathlib import Path
import random

from dotenv import load_dotenv
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================

API_KEY = os.getenv('GEMINI_API_KEY', '')
IS_PROD = os.getenv('NODE_ENV') == 'production'
REQUEST_TIMEOUT = 30
OUTPUT_DIR = './html_outputs'
METADATA_DIR = './metadata_outputs'

if not API_KEY:
    raise ValueError('GEMINI_API_KEY is not set in the environment variables.')

# Configure Gemini
genai.configure(api_key=API_KEY)

# Initialize FastAPI
app = FastAPI(
    title="Third-Party Site Detector API",
    description="Detects and classifies websites as third-party, competitor, or special URLs. Returns raw HTML and metadata for all processable sites.",
    version="1.0.0"
)

# ==================== PYDANTIC MODELS ====================

class DetectionInputModel(BaseModel):
    text: str
    url: str
    raw_url: Optional[str] = None
    snippet: Optional[str] = None
    position: Optional[int] = None
    related_to: Optional[str] = None
    highlight_fragment: Optional[str] = None
    related_claim: Optional[str] = None
    extraction_order: Optional[int] = None


class WebsiteMetadataModel(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    author: Optional[str] = None
    canonical_url: Optional[str] = None
    language: Optional[str] = None
    favicon: Optional[str] = None
    og_title: Optional[str] = None
    og_description: Optional[str] = None
    og_image: Optional[str] = None
    og_url: Optional[str] = None
    og_type: Optional[str] = None
    og_site_name: Optional[str] = None
    twitter_card: Optional[str] = None
    twitter_title: Optional[str] = None
    twitter_description: Optional[str] = None
    twitter_image: Optional[str] = None
    twitter_site: Optional[str] = None
    robots: Optional[str] = None
    viewport: Optional[str] = None
    charset: Optional[str] = None
    generator: Optional[str] = None
    theme_color: Optional[str] = None
    application_name: Optional[str] = None


class SpecialUrlInfo(BaseModel):
    url_type: str
    classification: str
    description: str
    platform: Optional[str] = None
    search_query: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    reasoning: str


class DetectionOutputModel(BaseModel):
    classification: Literal["third_party", "competitor", "special_url"]
    text: str
    url: str
    raw_url: str
    
    # For both competitor and third-party sites (raw HTML)
    html: Optional[str] = None
    metadata: Optional[WebsiteMetadataModel] = None
    
    # For all processable sites
    confidence_score: Optional[float] = None
    reasoning: Optional[str] = None
    
    # For special URLs
    special_url_info: Optional[SpecialUrlInfo] = None
    
    # Optional fields
    related_claim: Optional[str] = None
    html_file_path: Optional[str] = None
    metadata_file_path: Optional[str] = None


class BatchDetectionInput(BaseModel):
    inputs: List[DetectionInputModel]


# ==================== DATA CLASSES ====================

@dataclass
class DetectionInput:
    text: str
    url: str
    raw_url: Optional[str] = None
    snippet: Optional[str] = None
    position: Optional[int] = None
    related_to: Optional[str] = None
    highlight_fragment: Optional[str] = None
    related_claim: Optional[str] = None
    extraction_order: Optional[int] = None


@dataclass
class WebsiteMetadata:
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    author: Optional[str] = None
    canonical_url: Optional[str] = None
    language: Optional[str] = None
    favicon: Optional[str] = None
    og_title: Optional[str] = None
    og_description: Optional[str] = None
    og_image: Optional[str] = None
    og_url: Optional[str] = None
    og_type: Optional[str] = None
    og_site_name: Optional[str] = None
    twitter_card: Optional[str] = None
    twitter_title: Optional[str] = None
    twitter_description: Optional[str] = None
    twitter_image: Optional[str] = None
    twitter_site: Optional[str] = None
    robots: Optional[str] = None
    viewport: Optional[str] = None
    charset: Optional[str] = None
    generator: Optional[str] = None
    theme_color: Optional[str] = None
    application_name: Optional[str] = None


@dataclass
class GeminiClassificationResult:
    is_third_party: bool
    confidence_score: float
    reasoning: str


# ==================== UTILITY FUNCTIONS ====================

def log_debug(message: str):
    """Log debug messages only in non-production mode"""
    if not IS_PROD:
        print(message)


def clean_html_for_aeo(html_content: str) -> str:
    """
    Cleans HTML content for AEO/GEO (Answer Engine Optimization / Generative Engine Optimization) analysis
    by removing visual noise and preserving semantic structure and critical metadata.
    
    Preserves:
    - Structured data (JSON-LD, Microdata)
    - Semantic HTML (headings, paragraphs, lists, tables)
    - Important meta tags (description, article metadata, Open Graph)
    - Content hierarchy and structure
    - Alt text from images (for context)
    - Blockquotes and citations
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. Remove visual/interactive elements that don't contribute to content understanding
    for tag in soup.find_all(['style', 'noscript', 'iframe', 'form', 'button', 'input', 'select', 'textarea']):
        tag.decompose()
    
    # 2. Remove SVG but keep semantic icons if they have aria-label or title
    for svg in soup.find_all('svg'):
        try:
            aria_label = svg.get('aria-label', '').strip() if svg.get('aria-label') else ''
            title_tag = svg.find('title')
            title_text = title_tag.get_text().strip() if title_tag else ''
            
            if aria_label or title_text:
                svg.replace_with(f' [Icon: {aria_label or title_text}] ')
            else:
                svg.decompose()
        except (AttributeError, TypeError):
            svg.decompose()
    
    # 3. Keep scripts ONLY if they are JSON-LD structured data (critical for SEO/AEO)
    for script in soup.find_all('script'):
        try:
            script_type = script.get('type', '') if hasattr(script, 'get') else ''
            if script_type not in ['application/ld+json', 'application/json+ld']:
                script.decompose()
        except (AttributeError, TypeError):
            script.decompose()
    
    # 4. Handle Images - Keep alt text for context, remove the actual image
    for img in soup.find_all('img'):
        try:
            alt_text = img.get('alt', '').strip() if img.get('alt') else ''
            title_text = img.get('title', '').strip() if img.get('title') else ''
            
            if alt_text or title_text:
                replacement = f' [Image: {alt_text or title_text}] '
                img.replace_with(replacement)
            else:
                img.decompose()
        except (AttributeError, TypeError):
            img.decompose()
    
    # 5. Remove navigational elements (they don't contribute to main content)
    for tag in soup.find_all(['nav', 'footer']):
        tag.decompose()
    
    # 6. Keep header but only if it contains important content (h1, h2, etc)
    for header in soup.find_all('header'):
        try:
            # If header contains important headings, keep it; otherwise remove
            if not header.find(['h1', 'h2', 'h3']):
                header.decompose()
        except (AttributeError, TypeError):
            header.decompose()
    
    # 7. Filter Meta tags - Keep only SEO/AEO relevant ones
    for meta in soup.find_all('meta'):
        try:
            name = meta.get('name', '').lower() if meta.get('name') else ''
            property_attr = meta.get('property', '').lower() if meta.get('property') else ''
            
            # Keep important meta tags
            important_names = ['description', 'keywords', 'author', 'robots', 'article:published_time', 
                              'article:modified_time', 'article:author', 'article:section', 'article:tag']
            important_properties = ['og:title', 'og:description', 'og:type', 'og:url', 'og:image',
                                   'og:site_name', 'article:published_time', 'article:modified_time',
                                   'article:author', 'article:section', 'article:tag']
            
            is_important = (
                name in important_names or
                property_attr in important_properties or
                property_attr.startswith('article:')
            )
            
            if not is_important:
                meta.decompose()
        except (AttributeError, TypeError):
            meta.decompose()
    
    # 8. Filter Link tags - Keep canonical, alternate, and structured data links
    for link in soup.find_all('link'):
        try:
            rel = link.get('rel', []) if hasattr(link, 'get') else []
            rel_str = ' '.join(rel).lower() if isinstance(rel, list) else str(rel).lower()
            
            keep_links = ['canonical', 'alternate', 'amphtml', 'prev', 'next']
            
            if not any(keep_rel in rel_str for keep_rel in keep_links):
                link.decompose()
        except (AttributeError, TypeError):
            link.decompose()
    
    # 9. Define valid semantic tags that are important for content structure and AEO
    valid_tags: Set[str] = {
        # Headings (critical for content hierarchy)
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        # Text content
        'p', 'span', 'div', 'section', 'article', 'main', 'aside',
        # Lists (important for structured content)
        'ul', 'ol', 'li', 'dl', 'dt', 'dd',
        # Tables (structured data presentation)
        'table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th', 'caption',
        # Semantic text elements
        'strong', 'em', 'b', 'i', 'mark', 'small', 'del', 'ins', 'sub', 'sup',
        'blockquote', 'cite', 'q', 'abbr', 'code', 'pre', 'kbd', 'samp', 'var',
        # Structural elements
        'address', 'time', 'figure', 'figcaption',
        # Meta elements
        'title', 'meta', 'script', 'link', 'base',
        # Definition and details
        'details', 'summary',
        # Header (if it contains h1-h3)
        'header'
    }
    
    # 10. Clean attributes but preserve semantic ones
    # Create a list of tags to process to avoid modification during iteration
    all_tags = list(soup.find_all())
    
    for tag in all_tags:
        try:
            # Skip if tag has been removed from tree or is None
            if tag is None or not hasattr(tag, 'name') or tag.name is None:
                continue
                
            tag_name = tag.name.lower() if tag.name else ''
            
            if not tag_name or tag_name not in valid_tags:
                # Unwrap non-semantic tags but keep their content
                if hasattr(tag, 'unwrap'):
                    tag.unwrap()
            else:
                # Define allowed attributes per tag type
                allowed_attrs: Set[str] = set()
                
                if tag_name == 'script':
                    allowed_attrs = {'type', 'id'}
                elif tag_name == 'meta':
                    allowed_attrs = {'name', 'property', 'content', 'charset', 'http-equiv'}
                elif tag_name == 'link':
                    allowed_attrs = {'rel', 'href', 'hreflang', 'type'}
                elif tag_name == 'time':
                    allowed_attrs = {'datetime'}
                elif tag_name in ['article', 'section', 'div']:
                    allowed_attrs = {'id', 'itemscope', 'itemtype', 'itemprop'}  # Keep microdata
                elif tag_name == 'a':
                    allowed_attrs = {'href', 'rel'}
                elif tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    allowed_attrs = {'id'}  # Keep IDs for anchor links
                
                # Remove non-allowed attributes - check if attrs exists
                if hasattr(tag, 'attrs') and tag.attrs is not None:
                    attrs_to_remove = [attr for attr in list(tag.attrs.keys()) if attr not in allowed_attrs]
                    for attr in attrs_to_remove:
                        try:
                            del tag.attrs[attr]
                        except (KeyError, AttributeError):
                            pass
        except (AttributeError, TypeError, ValueError) as e:
            # Skip problematic tags
            log_debug(f'[HTML_CLEAN] Skipping tag due to error: {str(e)}')
            continue
    
    # 11. Get the cleaned HTML
    text = str(soup)
    
    # 12. Clean up excessive whitespace while preserving structure
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]
    
    return '\n'.join(lines)


# ==================== NEW GEMINI-BASED SPECIAL URL DETECTION ====================

def detect_special_url_with_gemini(url: str) -> Optional[Dict[str, Any]]:
    """
    Uses Gemini LLM to analyze the URL string (without fetching HTML) to determine
    if it is a special utility URL (Search Result, Social Share, Ad, etc.) 
    vs a processable Content URL.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""
        You are a URL structure expert. Analyze the specific string patterns of the URL below to determine if it is a "Special Utility URL" (which should be skipped) or a standard "Content URL" (which contains a webpage to be scraped).

        URL TO ANALYZE: {url}

        Categorize as "SPECIAL_URL" if it is:
        1. Search Engine Result Page (SERP): (e.g., google.com/search?, bing.com/search?, yahoo.com/search)
        2. Social Media Share/Intent Link: (e.g., twitter.com/intent/..., facebook.com/sharer/..., linkedin.com/shareArticle)
        3. Ad/Tracking/Redirect Link: (e.g., doubleclick.net, googleadservices.com, google.com/aclk)
        4. URL Shortener: (e.g., bit.ly, t.co, goo.gl)
        5. CDN/Static Asset: (e.g., ends in .jpg, .png, .pdf, or domains like cloudfront.net)
        6. System/Auth Page: (e.g., /login, /cart, /forgot-password)

        Categorize as "CONTENT_URL" if it is:
        - A standard homepage, blog post, product page, news article, or documentation page.

        Respond ONLY with valid JSON in this format:
        {{
            "is_special_url": boolean,
            "url_type": "search_engine_result" | "social_share_link" | "advertisement_link" | "url_shortener" | "cdn_asset_link" | "system_page" | "content_page",
            "classification": "special_url" (if is_special_url is true) or "content" (if false),
            "platform": "String (e.g. 'Google', 'Twitter', 'AWS') or null",
            "description": "Brief description of what this URL represents",
            "reasoning": "Why you classified it this way based on the URL structure/params"
        }}
        """

        log_debug(f'[SPECIAL_URL] Analyzing URL structure with Gemini: {url}')
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        
        text = response.text
        
        try:
            cleaned = re.sub(r'```json\n?|\n?```', '', text).strip()
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                log_debug('[SPECIAL_URL] Failed to parse Gemini response')
                return None

        if not parsed.get('is_special_url', False):
            return None

        return {
            'url_type': parsed.get('url_type', 'unknown_special'),
            'classification': 'special_url',
            'description': parsed.get('description', 'Special utility URL detected'),
            'platform': parsed.get('platform'),
            'search_query': None, 
            'parameters': {}, 
            'reasoning': parsed.get('reasoning', '')
        }

    except Exception as e:
        log_debug(f"[SPECIAL_URL_DETECTION] Error with Gemini: {str(e)}")
        return None


def is_valid_website_url(url: str) -> bool:
    """Validates if a URL points to an actual website"""
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        
        if not hostname:
            return False
        
        if parsed_url.scheme not in ['http', 'https']:
            return False
        
        return True
    except Exception:
        return False


def extract_clean_url(input_data: DetectionInput) -> str:
    """Extracts the clean URL from various input formats"""
    if input_data.raw_url:
        return input_data.raw_url
    
    if 'google.com' in input_data.url and 'url=' in input_data.url:
        try:
            query_string = input_data.url.split('?', 1)[1] if '?' in input_data.url else ''
            url_params = parse_qs(query_string)
            extracted_url = url_params.get('url', [None])[0] or url_params.get('q', [None])[0]
            if extracted_url and is_valid_website_url(extracted_url):
                return extracted_url
        except Exception:
            pass
    
    return input_data.url


def normalize_text(text: str) -> str:
    """Normalizes text by removing extra whitespace"""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]
    return '\n'.join(lines).strip()


def ensure_output_dir(directory: str) -> None:
    """Creates output directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_html_to_file(html: str, url: str, raw: bool) -> str:
    """Saves HTML to a file and returns the file path"""
    ensure_output_dir(OUTPUT_DIR)
    
    parsed_url = urlparse(url)
    hostname = re.sub(r'[^a-z0-9]', '_', parsed_url.hostname or 'unknown', flags=re.IGNORECASE)
    timestamp = int(time.time() * 1000)
    
    filename = f'raw_{hostname}_{timestamp}.html' if raw else f'{hostname}_{timestamp}.html'
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    log_debug(f'[HTML] Saved to: {file_path}')
    return file_path


def save_metadata_to_file(metadata: WebsiteMetadata, url: str) -> str:
    """Saves metadata to a JSON file and returns the file path"""
    ensure_output_dir(METADATA_DIR)
    
    parsed_url = urlparse(url)
    hostname = re.sub(r'[^a-z0-9]', '_', parsed_url.hostname or 'unknown', flags=re.IGNORECASE)
    timestamp = int(time.time() * 1000)
    filename = f'{hostname}_{timestamp}_metadata.json'
    file_path = os.path.join(METADATA_DIR, filename)
    
    metadata_dict = asdict(metadata)
    metadata_dict = {k: v for k, v in metadata_dict.items() if v is not None and v != [] and v != ''}
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2)
    
    log_debug(f'[METADATA] Saved to: {file_path}')
    return file_path


def save_api_output_to_file(output: Any, url: Optional[str] = None) -> str:
    """Saves the entire API output (detection result) to a JSON file and returns the file path"""
    ensure_output_dir(METADATA_DIR)

    # Try to obtain a URL for naming; fallback to provided url or 'unknown'
    out_url = None
    try:
        # If it's a Pydantic model, it will have .dict()
        if hasattr(output, 'raw_url') and output.raw_url:
            out_url = output.raw_url
        elif hasattr(output, 'url') and output.url:
            out_url = output.url
        elif url:
            out_url = url
    except Exception:
        out_url = url

    parsed_url = urlparse(out_url or 'unknown')
    hostname = re.sub(r'[^a-z0-9]', '_', parsed_url.hostname or 'unknown', flags=re.IGNORECASE)
    timestamp = int(time.time() * 1000)
    filename = f'{hostname}_{timestamp}_api_output.json'
    file_path = os.path.join(METADATA_DIR, filename)

    # Convert output to serializable dict
    try:
        if hasattr(output, 'dict'):
            data = output.dict()
        elif hasattr(output, '__dict__'):
            data = output.__dict__
        else:
            data = output
    except Exception:
        data = str(output)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log_debug(f'[API_OUTPUT] Saved to: {file_path}')
    return file_path


# ==================== METADATA EXTRACTION ====================

def extract_metadata(html: str, url: str) -> WebsiteMetadata:
    """Extracts comprehensive metadata from HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    
    def get_meta(selector: str, attr: str = 'content') -> Optional[str]:
        tag = soup.select_one(selector)
        if tag:
            content = tag.get(attr, '').strip()
            return content if content else None
        return None
    
    def get_meta_array(selector: str) -> List[str]:
        content = get_meta(selector)
        if not content:
            return []
        return [k.strip() for k in content.split(',') if k.strip()]
    
    favicon: Optional[str] = None
    favicon_selectors = [
        'link[rel="icon"]',
        'link[rel="shortcut icon"]',
        'link[rel="apple-touch-icon"]',
        'link[rel="apple-touch-icon-precomposed"]'
    ]
    
    for selector in favicon_selectors:
        tag = soup.select_one(selector)
        if tag:
            href = tag.get('href')
            if href:
                try:
                    favicon = urljoin(url, href)
                    break
                except Exception:
                    continue
    
    canonical_url: Optional[str] = None
    canonical_tag = soup.select_one('link[rel="canonical"]')
    if canonical_tag:
        canonical_href = canonical_tag.get('href')
        if canonical_href:
            try:
                canonical_url = urljoin(url, canonical_href)
            except Exception:
                canonical_url = canonical_href
    
    title_tag = soup.find('title')
    title = title_tag.get_text().strip() if title_tag else None
    if not title:
        title = get_meta('meta[property="og:title"]') or get_meta('meta[name="twitter:title"]')
    
    html_tag = soup.find('html')
    language = html_tag.get('lang') if html_tag else None
    if not language:
        language = get_meta('meta[http-equiv="content-language"]')
    
    charset_tag = soup.find('meta', charset=True)
    charset = charset_tag.get('charset') if charset_tag else None
    if not charset:
        charset = get_meta('meta[http-equiv="Content-Type"]')
    
    metadata = WebsiteMetadata(
        title=title,
        description=get_meta('meta[name="description"]') or get_meta('meta[property="og:description"]'),
        keywords=get_meta_array('meta[name="keywords"]'),
        author=get_meta('meta[name="author"]'),
        canonical_url=canonical_url,
        language=language,
        favicon=favicon,
        og_title=get_meta('meta[property="og:title"]'),
        og_description=get_meta('meta[property="og:description"]'),
        og_image=get_meta('meta[property="og:image"]'),
        og_url=get_meta('meta[property="og:url"]'),
        og_type=get_meta('meta[property="og:type"]'),
        og_site_name=get_meta('meta[property="og:site_name"]'),
        twitter_card=get_meta('meta[name="twitter:card"]'),
        twitter_title=get_meta('meta[name="twitter:title"]'),
        twitter_description=get_meta('meta[name="twitter:description"]'),
        twitter_image=get_meta('meta[name="twitter:image"]'),
        twitter_site=get_meta('meta[name="twitter:site"]'),
        robots=get_meta('meta[name="robots"]'),
        viewport=get_meta('meta[name="viewport"]'),
        charset=charset,
        generator=get_meta('meta[name="generator"]'),
        theme_color=get_meta('meta[name="theme-color"]'),
        application_name=get_meta('meta[name="application-name"]') or get_meta('meta[property="al:ios:app_name"]')
    )
    
    return metadata


# ==================== SCRAPING FUNCTIONS ====================

def fetch_raw_html(url: str) -> str:
    """Fetches raw HTML with anti-blocking measures"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
    ]
    
    random_user_agent = random.choice(user_agents)
    
    headers = {
        'User-Agent': random_user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        return response.text
    except requests.HTTPError as e:
        if e.response.status_code == 403:
            parsed_url = urlparse(url)
            retry_headers = headers.copy()
            retry_headers['User-Agent'] = random.choice(user_agents)
            retry_headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            response = requests.get(url, headers=retry_headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            response.raise_for_status()
            return response.text
        raise


def extract_content_for_analysis(html: str) -> str:
    """Extracts meaningful content from HTML for analysis"""
    soup = BeautifulSoup(html, 'html.parser')
    
    for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript', 'svg']):
        tag.decompose()
    
    title_tag = soup.find('title')
    title = title_tag.get_text() if title_tag else ''
    
    h1_tag = soup.find('h1')
    if not title and h1_tag:
        title = h1_tag.get_text()
    
    meta_desc_tag = soup.find('meta', attrs={'name': 'description'}) or \
                    soup.find('meta', attrs={'property': 'og:description'})
    meta_description = meta_desc_tag.get('content', '') if meta_desc_tag else ''
    
    main_selectors = ['main', 'article', '[role="main"]', '.content', '#content', '.main']
    main_content = ''
    
    for selector in main_selectors:
        element = soup.select_one(selector)
        if element:
            main_content = element.get_text()
            break
    
    if not main_content:
        body_tag = soup.find('body')
        main_content = body_tag.get_text() if body_tag else ''
    
    headings = []
    for tag in soup.find_all(['h1', 'h2', 'h3']):
        text = tag.get_text().strip()
        if text and len(text) < 200:
            headings.append(text)
    
    combined = f"""
TITLE: {title}
META DESCRIPTION: {meta_description}
HEADINGS: {' | '.join(headings[:10])}
CONTENT PREVIEW: {normalize_text(main_content)[:3000]}
    """.strip()
    
    return combined


# ==================== GEMINI AI CLASSIFICATION ====================

def classify_with_gemini(content: str, url: str, search_text: str) -> GeminiClassificationResult:
    """Calls Gemini to classify if the site is third-party or competitor (own product)"""
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    prompt = f"""
You are analyzing a website to determine if it is a THIRD-PARTY review/comparison/listing site or the OFFICIAL PRODUCT/COMPETITOR site.

SEARCH CONTEXT: The user searched for "{search_text}"
WEBSITE URL: {url}

WEBSITE CONTENT:
{content}

Please analyze this website and determine:
1. Is this a THIRD-PARTY site (review site, comparison site, listing site, blog, news site, etc.) OR is it the OFFICIAL product/company site (COMPETITOR)?

THIRD-PARTY indicators:
- Reviews or compares multiple products/services
- Lists or ranks different solutions
- Blog posts about industry topics
- News articles or press releases
- Affiliate links or "Best of" lists
- Generic domain names (e.g., "reviews.com", "best-software.com")
- Multiple vendor/product mentions

COMPETITOR (Own Product) indicators:
- Official company website
- Single product/service focus
- Pricing pages, sign-up forms, demos
- "About Us", "Contact", "Careers" pages for that specific company
- Official product documentation
- Company-specific branding throughout

Respond ONLY with valid JSON in this exact format (no markdown, no backticks):
{{
  "is_third_party": true or false,
  "confidence_score": 0.0 to 1.0,
  "reasoning": "Brief explanation of your classification"
}}
"""
    
    log_debug('[GEMINI] Sending classification request...')
    
    response = model.generate_content(prompt)
    text = response.text
    
    try:
        cleaned = re.sub(r'```json\n?|\n?```', '', text).strip()
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            parsed = json.loads(json_match.group(0))
        else:
            raise ValueError('Failed to parse Gemini response as JSON')
    
    return GeminiClassificationResult(
        is_third_party=parsed['is_third_party'],
        confidence_score=parsed['confidence_score'],
        reasoning=parsed['reasoning']
    )


# ==================== MAIN DETECTION FUNCTION ====================

def detect_third_party_site(input_data: DetectionInput) -> Optional[DetectionOutputModel]:
    """Main function to detect and classify website"""
    try:
        clean_url = extract_clean_url(input_data)
        
        log_debug(f'\n[DETECTOR] Processing: {input_data.text}')
        log_debug(f'[DETECTOR] Original URL: {input_data.url}')
        log_debug(f'[DETECTOR] Clean URL: {clean_url}')
        
        # 1. SPECIAL URL CHECK (LLM BASED)
        special_url_info = detect_special_url_with_gemini(clean_url)
        
        if special_url_info:
            log_debug(f'[DETECTOR] Detected special URL type: {special_url_info["url_type"]}')
            
            return DetectionOutputModel(
                classification="special_url",
                text=input_data.text,
                url=input_data.url,
                raw_url=clean_url,
                related_claim=input_data.related_claim,
                special_url_info=SpecialUrlInfo(**special_url_info)
            )
        
        # 2. VALIDATION
        if not is_valid_website_url(clean_url):
            log_debug('[DETECTOR] Skipping invalid/non-website URL')
            return None
        
        # 3. FETCH HTML
        log_debug('[DETECTOR] Not a special URL. Fetching HTML...')
        html = fetch_raw_html(clean_url)
        log_debug(f'[DETECTOR] HTML fetched: {len(html)} characters')
        
        # 4. Extract content for analysis
        content = extract_content_for_analysis(html)
        
        # 5. Classify with Gemini
        log_debug('[DETECTOR] Classifying with Gemini...')
        classification = classify_with_gemini(content, clean_url, input_data.text)
        
        # 6. UNIFIED PROCESSING FOR BOTH THIRD-PARTY AND COMPETITOR
        # Both types now get metadata extraction and raw HTML
        
        classification_type = "third_party" if classification.is_third_party else "competitor"
        log_debug(f'[DETECTOR] Classified as {classification_type.upper()} site')
        
        # Extract metadata
        log_debug('[DETECTOR] Extracting metadata...')
        metadata = extract_metadata(html, clean_url)
        
        # Save files (optional in production)
        html_file_path = save_html_to_file(html, clean_url, True) if not IS_PROD else None
        metadata_file_path = save_metadata_to_file(metadata, clean_url) if not IS_PROD else None
        
        # Convert metadata to Pydantic model
        metadata_dict = asdict(metadata)
        metadata_model = WebsiteMetadataModel(**metadata_dict)
        
        log_debug('[DETECTOR] Result:')
        log_debug(f'  Classification: {classification_type.upper()}')
        log_debug(f'  Confidence: {classification.confidence_score}')
        log_debug(f'  Reasoning: {classification.reasoning}')
        log_debug(f'  HTML size: {len(html)} characters')
        if not IS_PROD:
            log_debug(f'  HTML saved: {html_file_path}')
            log_debug(f'  Metadata saved: {metadata_file_path}')
        
        return DetectionOutputModel(
            classification=classification_type,
            text=input_data.text,
            url=input_data.url,
            raw_url=clean_url,
            related_claim=input_data.related_claim,
            html=html,
            metadata=metadata_model,
            confidence_score=classification.confidence_score,
            reasoning=classification.reasoning,
            html_file_path=html_file_path,
            metadata_file_path=metadata_file_path
        )
            
    except Exception as error:
        log_debug(f'[DETECTOR] Error: {str(error)}')
        raise


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Third-Party Site Detector API",
        "version": "1.0.0",
        "description": "Detects and classifies websites as third-party, competitor, or special URLs. Returns raw HTML and metadata for all processable sites.",
        "endpoints": {
            "/detect": "POST - Detect single website",
            "/detect/batch": "POST - Detect multiple websites",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "environment": "production" if IS_PROD else "development"}


@app.post("/detect", response_model=DetectionOutputModel)
async def detect_endpoint(input_data: DetectionInputModel):
    """
    Detect and classify a single website
    
    Returns:
    - classification: "third_party", "competitor", or "special_url"
    - For both competitor and third-party sites: includes raw html and metadata
    - For special URLs: detailed information about the URL type
    """
    try:
        detection_input = DetectionInput(
            text=input_data.text,
            url=input_data.url,
            raw_url=input_data.raw_url,
            snippet=input_data.snippet,
            position=input_data.position,
            related_to=input_data.related_to,
            highlight_fragment=input_data.highlight_fragment,
            related_claim=input_data.related_claim,
            extraction_order=input_data.extraction_order
        )
        
        result = detect_third_party_site(detection_input)
        
        if result is None:
            raise HTTPException(status_code=400, detail="Invalid or unprocessable URL")
        # Save API output to file (non-production only to mirror HTML/metadata saving behavior)
        try:
            if not IS_PROD:
                save_api_output_to_file(result)
        except Exception as e:
            log_debug(f'[API_OUTPUT] Failed to save API output: {str(e)}')

        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/detect/batch")
async def detect_batch_endpoint(batch_input: BatchDetectionInput):
    """
    Detect and classify multiple websites in batch
    
    Returns a list of detection results for each input URL
    """
    results = []
    errors = []
    
    for idx, input_data in enumerate(batch_input.inputs):
        try:
            detection_input = DetectionInput(
                text=input_data.text,
                url=input_data.url,
                raw_url=input_data.raw_url,
                snippet=input_data.snippet,
                position=input_data.position,
                related_to=input_data.related_to,
                highlight_fragment=input_data.highlight_fragment,
                related_claim=input_data.related_claim,
                extraction_order=input_data.extraction_order
            )
            
            result = detect_third_party_site(detection_input)
            
            if result:
                results.append(result)
                # Save API output for each successful result
                try:
                    if not IS_PROD:
                        save_api_output_to_file(result)
                except Exception as e:
                    log_debug(f'[API_OUTPUT] Failed to save batch API output: {str(e)}')
            else:
                errors.append({
                    "index": idx,
                    "url": input_data.url,
                    "error": "Invalid or unprocessable URL"
                })
            
            # Rate limiting between requests
            time.sleep(1)
            
        except Exception as e:
            errors.append({
                "index": idx,
                "url": input_data.url,
                "error": str(e)
            })
    
    return {
        "results": results,
        "errors": errors,
        "total_processed": len(batch_input.inputs),
        "successful": len(results),
        "failed": len(errors)
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)