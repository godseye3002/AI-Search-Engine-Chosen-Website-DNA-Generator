"""
Classification Core Module

Refactored core logic from website_classification_claude_raw_html.py
to be callable as a function for pipeline integration.
"""

import os
import json
import re
import time
import random
from typing import Optional, Dict, Any, Literal
from urllib.parse import urlparse, parse_qs, urljoin
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv('GEMINI_API_KEY', '')
REQUEST_TIMEOUT = 30

if not API_KEY:
    raise ValueError('GEMINI_API_KEY is not set in the environment variables.')

# Configure Gemini
genai.configure(api_key=API_KEY)


class ClassificationResult:
    """Result of website classification"""
    
    def __init__(
        self,
        classification: Literal["third_party", "competitor", "special_url"],
        text: str,
        url: str,
        raw_url: str,
        html: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None,
        reasoning: Optional[str] = None,
        special_url_info: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        self.classification = classification
        self.text = text
        self.url = url
        self.raw_url = raw_url
        self.html = html
        self.metadata = metadata
        self.confidence_score = confidence_score
        self.reasoning = reasoning
        self.special_url_info = special_url_info
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'classification': self.classification,
            'text': self.text,
            'url': self.url,
            'raw_url': self.raw_url
        }
        
        if self.html is not None:
            result['html'] = self.html
        if self.metadata is not None:
            result['metadata'] = self.metadata
        if self.confidence_score is not None:
            result['confidence_score'] = self.confidence_score
        if self.reasoning is not None:
            result['reasoning'] = self.reasoning
        if self.special_url_info is not None:
            result['special_url_info'] = self.special_url_info
        if self.error is not None:
            result['error'] = self.error
            
        return result


def detect_special_url_with_gemini(url: str) -> Optional[Dict[str, Any]]:
    """
    Uses Gemini LLM to analyze the URL string to determine if it is a special utility URL.
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


def extract_clean_url(input_data: Dict[str, Any]) -> str:
    """Extracts the clean URL from various input formats"""
    url = input_data.get('url', '')
    raw_url = input_data.get('raw_url')
    
    if raw_url:
        return raw_url
    
    if 'google.com' in url and 'url=' in url:
        try:
            query_string = url.split('?', 1)[1] if '?' in url else ''
            url_params = parse_qs(query_string)
            extracted_url = url_params.get('url', [None])[0] or url_params.get('q', [None])[0]
            if extracted_url and is_valid_website_url(extracted_url):
                return extracted_url
        except Exception:
            pass
    
    return url


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


def extract_metadata(html: str, url: str) -> Dict[str, Any]:
    """Extracts comprehensive metadata from HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    
    def get_meta(selector: str, attr: str = 'content') -> Optional[str]:
        tag = soup.select_one(selector)
        if tag:
            content = tag.get(attr, '').strip()
            return content if content else None
        return None
    
    def get_meta_array(selector: str) -> list:
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
    
    metadata = {
        'title': title,
        'description': get_meta('meta[name="description"]') or get_meta('meta[property="og:description"]'),
        'keywords': get_meta_array('meta[name="keywords"]'),
        'author': get_meta('meta[name="author"]'),
        'canonical_url': canonical_url,
        'language': language,
        'favicon': favicon,
        'og_title': get_meta('meta[property="og:title"]'),
        'og_description': get_meta('meta[property="og:description"]'),
        'og_image': get_meta('meta[property="og:image"]'),
        'og_url': get_meta('meta[property="og:url"]'),
        'og_type': get_meta('meta[property="og:type"]'),
        'og_site_name': get_meta('meta[property="og:site_name"]'),
        'twitter_card': get_meta('meta[name="twitter:card"]'),
        'twitter_title': get_meta('meta[name="twitter:title"]'),
        'twitter_description': get_meta('meta[name="twitter:description"]'),
        'twitter_image': get_meta('meta[name="twitter:image"]'),
        'twitter_site': get_meta('meta[name="twitter:site"]'),
        'robots': get_meta('meta[name="robots"]'),
        'viewport': get_meta('meta[name="viewport"]'),
        'charset': charset,
        'generator': get_meta('meta[name="generator"]'),
        'theme_color': get_meta('meta[name="theme-color"]'),
        'application_name': get_meta('meta[name="application-name"]') or get_meta('meta[property="al:ios:app_name"]')
    }
    
    # Remove None values
    return {k: v for k, v in metadata.items() if v is not None and v != [] and v != ''}


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
    
    # Normalize text
    def normalize_text(text: str) -> str:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return '\n'.join(lines).strip()
    
    combined = f"""
TITLE: {title}
META DESCRIPTION: {meta_description}
HEADINGS: {' | '.join(headings[:10])}
CONTENT PREVIEW: {normalize_text(main_content)[:3000]}
    """.strip()
    
    return combined


def classify_with_gemini(content: str, url: str, search_text: str) -> Dict[str, Any]:
    """Calls Gemini to classify if the site is third-party or competitor"""
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
    
    return parsed


def classify_website(input_data: Dict[str, Any]) -> ClassificationResult:
    """
    Main function to classify a single website.
    
    Args:
        input_data: Dictionary containing url, text, and other metadata
        
    Returns:
        ClassificationResult with the classification and metadata
    """
    try:
        # Extract clean URL
        clean_url = extract_clean_url(input_data)
        
        # Check for special URL first
        special_url_info = detect_special_url_with_gemini(clean_url)
        
        if special_url_info:
            return ClassificationResult(
                classification="special_url",
                text=input_data.get('text', ''),
                url=input_data.get('url', ''),
                raw_url=clean_url,
                special_url_info=special_url_info
            )
        
        # Validate URL
        if not is_valid_website_url(clean_url):
            return ClassificationResult(
                classification="special_url",  # Treat invalid URLs as special
                text=input_data.get('text', ''),
                url=input_data.get('url', ''),
                raw_url=clean_url,
                error="Invalid or non-website URL"
            )
        
        # Fetch HTML
        try:
            html = fetch_raw_html(clean_url)
        except Exception as e:
            return ClassificationResult(
                classification="special_url",
                text=input_data.get('text', ''),
                url=input_data.get('url', ''),
                raw_url=clean_url,
                error=f"Failed to fetch HTML: {str(e)}"
            )
        
        # Extract metadata
        metadata = extract_metadata(html, clean_url)
        
        # Extract content for analysis
        content = extract_content_for_analysis(html)
        
        # Classify with Gemini
        try:
            classification_result = classify_with_gemini(
                content, 
                clean_url, 
                input_data.get('related_to', '')
            )
        except Exception as e:
            return ClassificationResult(
                classification="special_url",
                text=input_data.get('text', ''),
                url=input_data.get('url', ''),
                raw_url=clean_url,
                html=html,
                metadata=metadata,
                error=f"Classification failed: {str(e)}"
            )
        
        # Determine final classification
        is_third_party = classification_result.get('is_third_party', True)
        final_classification = "third_party" if is_third_party else "competitor"
        
        return ClassificationResult(
            classification=final_classification,
            text=input_data.get('text', ''),
            url=input_data.get('url', ''),
            raw_url=clean_url,
            html=html,
            metadata=metadata,
            confidence_score=classification_result.get('confidence_score'),
            reasoning=classification_result.get('reasoning')
        )
        
    except Exception as e:
        return ClassificationResult(
            classification="special_url",
            text=input_data.get('text', ''),
            url=input_data.get('url', ''),
            raw_url=input_data.get('raw_url', ''),
            error=f"Unexpected error: {str(e)}"
        )


if __name__ == "__main__":
    # Test the classification function
    test_input = {
        'url': 'https://example.com',
        'text': 'Example Website',
        'related_to': 'test search'
    }
    
    result = classify_website(test_input)
    print("Classification result:")
    print(json.dumps(result.to_dict(), indent=2))
