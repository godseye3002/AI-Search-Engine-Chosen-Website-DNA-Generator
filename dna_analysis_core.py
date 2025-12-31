"""
DNA Analysis Core

Refactored forensic analysis logic for Stage 2 of the pipeline.
Extracts winning content DNA from classified HTML data.
"""

import os
import json
import re
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv() 
API_KEY = os.getenv('GEMINI_API_KEY', '')
genai.configure(api_key=API_KEY)

# Setup Gemini API tracker
gemini_logger = logging.getLogger('gemini_api')
gemini_logger.setLevel(logging.INFO)
if not gemini_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[GEMINI] %(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    gemini_logger.addHandler(handler)

def track_gemini_call(func_name: str, prompt: str, response: Any = None, error: Exception = None, start_time: float = None):
    """Track Gemini API calls for debugging"""
    end_time = time.time()
    duration = end_time - start_time if start_time else 0
    
    gemini_logger.info(f"=== {func_name} ===")
    gemini_logger.info(f"Duration: {duration:.2f}s")
    gemini_logger.info(f"Prompt length: {len(prompt)} chars")
    
    if error:
        gemini_logger.error(f"ERROR: {str(error)}")
        gemini_logger.error(f"Error type: {type(error).__name__}")
    elif response:
        if hasattr(response, 'text'):
            gemini_logger.info(f"Response length: {len(response.text)} chars")
            gemini_logger.info(f"Response preview: {response.text[:200]}...")
        else:
            gemini_logger.info(f"Response type: {type(response)}")
            gemini_logger.info(f"Response preview: {str(response)[:200]}...")
    
    gemini_logger.info("=" * 50)


@dataclass
class DNAAnalysisResult:
    """Result of DNA analysis for a single job"""
    job_id: str
    url: str
    classification: str
    dna_profile: Dict[str, Any]
    causal_evidence: List[Dict[str, Any]]
    content_insights: Dict[str, Any]
    error: Optional[str] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'job_id': self.job_id,
            'url': self.url,
            'classification': self.classification,
            'dna_profile': self.dna_profile,
            'causal_evidence': self.causal_evidence,
            'content_insights': self.content_insights,
            'error': self.error,
            'processing_time': self.processing_time,
            'analysis_timestamp': datetime.now().isoformat()
        }


class SemanticHTMLChunker:
    """Smart HTML chunking preserving DOM structure with overlaps"""
    
    def __init__(self, max_tokens_per_chunk: int = 15000, overlap_tokens: int = 50):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
    
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def get_dom_path(self, element) -> str:
        """Generate DOM signature"""
        path = []
        current = element
        while current and current.name:
            tag = current.name.upper()
            if current.get('class'):
                tag += f".{current['class'][0]}"
            path.insert(0, tag)
            current = current.parent
            if len(path) >= 5:
                break
        return ' > '.join(path)
    
    def chunk_html_with_structure(self, html: str) -> List[Dict[str, Any]]:
        """Create max 5 semantic chunks with DOM structure preserved"""
        soup = BeautifulSoup(html, 'html.parser')
        chunks = []
        
        chunk_1 = self._extract_metadata_chunk(soup)
        if chunk_1:
            chunks.append(chunk_1)
        
        chunk_2 = self._extract_structural_chunk(soup)
        if chunk_2:
            chunks.append(chunk_2)
        
        main_chunks = self._extract_main_content_with_overlap(soup)
        chunks.extend(main_chunks)
        
        chunk_5 = self._extract_sidebar_chunk(soup)
        if chunk_5:
            chunks.append(chunk_5)
        
        return chunks[:5]  # Limit to 5 chunks max
    
    def _extract_metadata_chunk(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        head = soup.find('head')
        if not head:
            return None
        
        metadata_html = str(head)
        json_ld = soup.find_all('script', type='application/ld+json')
        for script in json_ld:
            metadata_html += f"\n{str(script)}"
        
        structured = []
        for tag in ['title', 'meta']:
            for elem in head.find_all(tag):
                structured.append({
                    'tag': tag,
                    'html': str(elem),
                    'dom_path': self.get_dom_path(elem)
                })
        
        return {
            'chunk_id': 'chunk_1_metadata',
            'chunk_name': 'Metadata Layer',
            'raw_html': metadata_html[:self.max_tokens_per_chunk * 4],
            'structured_elements': structured,
            'section_type': 'metadata'
        }
    
    def _extract_structural_chunk(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        structural_parts = []
        
        for tag_name in ['nav', 'header', 'footer']:
            for elem in soup.find_all(tag_name):
                structural_parts.append({
                    'tag': tag_name,
                    'html': str(elem)[:2000],
                    'dom_path': self.get_dom_path(elem)
                })
        
        if not structural_parts:
            return None
        
        return {
            'chunk_id': 'chunk_2_structural',
            'chunk_name': 'Navigation & Structure',
            'raw_html': '\n'.join([part['html'] for part in structural_parts]),
            'structured_elements': structural_parts,
            'section_type': 'structural'
        }
    
    def _extract_main_content_with_overlap(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        main_chunks = []
        
        # Try to find main content area
        main_content = (soup.find('main') or 
                        soup.find('article') or 
                        soup.find('div', class_=re.compile(r'content|main|article')))
        
        if main_content:
            content_html = str(main_content)
            if self.estimate_tokens(content_html) > self.max_tokens_per_chunk:
                # Split large content
                chunks = self._split_large_content(main_content)
                main_chunks.extend(chunks)
            else:
                main_chunks.append({
                    'chunk_id': 'chunk_3_main',
                    'chunk_name': 'Main Content',
                    'raw_html': content_html,
                    'structured_elements': [],
                    'section_type': 'main_content'
                })
        
        return main_chunks
    
    def _extract_sidebar_chunk(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        sidebar = (soup.find('aside') or 
                  soup.find('div', class_=re.compile(r'sidebar|side')))
        
        if not sidebar:
            return None
        
        return {
            'chunk_id': 'chunk_5_sidebar',
            'chunk_name': 'Sidebar & Related',
            'raw_html': str(sidebar)[:self.max_tokens_per_chunk * 4],
            'structured_elements': [],
            'section_type': 'sidebar'
        }
    
    def _split_large_content(self, element) -> List[Dict[str, Any]]:
        """Split large content into smaller chunks"""
        chunks = []
        current_chunk = ""
        chunk_id = 3
        
        for child in element.children:
            if hasattr(child, 'name'):
                child_html = str(child)
                if self.estimate_tokens(current_chunk + child_html) > self.max_tokens_per_chunk:
                    if current_chunk:
                        chunks.append({
                            'chunk_id': f'chunk_{chunk_id}_main',
                            'chunk_name': f'Main Content Part {chunk_id-2}',
                            'raw_html': current_chunk,
                            'structured_elements': [],
                            'section_type': 'main_content'
                        })
                        current_chunk = child_html
                        chunk_id += 1
                    else:
                        # Single element too large, truncate
                        chunks.append({
                            'chunk_id': f'chunk_{chunk_id}_main',
                            'chunk_name': f'Main Content Part {chunk_id-2}',
                            'raw_html': child_html[:self.max_tokens_per_chunk * 4],
                            'structured_elements': [],
                            'section_type': 'main_content'
                        })
                        chunk_id += 1
                else:
                    current_chunk += child_html
        
        if current_chunk:
            chunks.append({
                'chunk_id': f'chunk_{chunk_id}_main',
                'chunk_name': f'Main Content Part {chunk_id-2}',
                'raw_html': current_chunk,
                'structured_elements': [],
                'section_type': 'main_content'
            })
        
        return chunks


class ForensicDNAAnalyzer:
    """AI-powered forensic DNA analyzer for content analysis"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.chunker = SemanticHTMLChunker()
    
    def analyze_content_dna(self, classified_data: Dict[str, Any], 
                          ai_response: Dict[str, Any]) -> DNAAnalysisResult:
        """
        Perform forensic DNA analysis on classified content.
        
        Args:
            classified_data: Output from Stage 1 classification
            ai_response: Original AI response with query and context
            
        Returns:
            DNAAnalysisResult with extracted DNA profile and evidence
        """
        start_time = datetime.now()
        
        try:
            # Extract key information
            url = classified_data.get('url', '')
            classification = classified_data.get('classification', '')
            html_content = classified_data.get('html', '')
            metadata = classified_data.get('metadata', {})
            
            # Generate chunks
            chunks = self.chunker.chunk_html_with_structure(html_content)
            
            # Prepare analysis context
            query = ai_response.get('query', '')
            ai_overview = ai_response.get('ai_overview', '')
            
            # Analyze each chunk
            chunk_analyses = []
            for chunk in chunks:
                analysis = self._analyze_chunk(chunk, query, ai_overview, classification)
                chunk_analyses.append(analysis)
            
            # Synthesize DNA profile
            dna_profile = self._synthesize_dna_profile(chunk_analyses, metadata)
            
            # Extract causal evidence
            causal_evidence = self._extract_causal_evidence(chunk_analyses, chunks)
            
            # Generate content insights
            content_insights = self._generate_content_insights(dna_profile, classification)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return DNAAnalysisResult(
                job_id=classified_data.get('job_id', ''),
                url=url,
                classification=classification,
                dna_profile=dna_profile,
                causal_evidence=causal_evidence,
                content_insights=content_insights,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return DNAAnalysisResult(
                job_id=classified_data.get('job_id', ''),
                url=classified_data.get('url', ''),
                classification=classified_data.get('classification', ''),
                dna_profile={},
                causal_evidence=[],
                content_insights={},
                error=str(e),
                processing_time=processing_time
            )
    
    def _analyze_chunk(self, chunk: Dict[str, Any], query: str, 
                      ai_overview: str, classification: str) -> Dict[str, Any]:
        """Analyze individual HTML chunk"""
        
        # Limit HTML content to avoid token limits
        html_content = chunk['raw_html']
        if len(html_content) > 4000:
            html_content = html_content[:4000] + "..."
        
        prompt = f"""
        **ROLE:** Forensic Content DNA Analyst
        **TASK:** Extract actionable insights from HTML content for competitive analysis.

        CONTEXT:
        - User Query: {query}
        - AI Overview: {ai_overview[:300]}...
        - Content Classification: {classification}

        HTML CHUNK TO ANALYZE:
        Chunk ID: {chunk['chunk_id']}
        Section Type: {chunk['section_type']}
        HTML Content: {html_content}

        **ANALYSIS REQUIREMENTS:**
        1. Identify entities (tools, platforms, metrics)
        2. Extract statistics and numbers
        3. Find unique value propositions
        4. Analyze content structure
        5. Determine relevance score

        **CRITICAL:** Return ONLY valid JSON. No explanations outside JSON.

        {{
            "relevance_score": 85,
            "key_elements": [
                {{
                    "element_type": "text_content",
                    "content_snippet": "Firebase stands out for tracking spending across different channels",
                    "relevance_reason": "Direct tool comparison with specific feature",
                    "ranking_factor": "entity_relevance"
                }},
                {{
                    "element_type": "heading",
                    "content_snippet": "Best Mobile Measurement Partners",
                    "relevance_reason": "Exact query match in heading",
                    "ranking_factor": "keyword_optimization"
                }}
            ],
            "content_patterns": [
                {{
                    "pattern": "comparison_table",
                    "frequency": "high",
                    "impact": "user_engagement"
                }}
            ],
            "semantic_density": {{
                "topic_relevance": 90,
                "entity_coverage": "Firebase, MMP, attribution, analytics",
                "content_depth": "medium"
            }}
        }}
        """
        
        start_time = None
        try:
            model = genai.GenerativeModel(
                model_name="gemini-2.5-pro",
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 40,
                    # "max_output_tokens": 2048,  # Reduced to avoid token limits
                }
            )
            
            # Track Gemini API call
            start_time = time.time()
            response = model.generate_content(prompt)
            track_gemini_call("DNA Analysis", prompt, response=response, start_time=start_time)
            
            response_text = response.text.strip()
            
            # Clean response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            
            # Parse JSON
            analysis = json.loads(response_text)
            
            # Validate required fields
            if not isinstance(analysis.get('relevance_score'), (int, float)):
                analysis['relevance_score'] = 50
            if not isinstance(analysis.get('key_elements'), list):
                analysis['key_elements'] = []
            if not isinstance(analysis.get('content_patterns'), list):
                analysis['content_patterns'] = []
            if not isinstance(analysis.get('semantic_density'), dict):
                analysis['semantic_density'] = {"topic_relevance": 50, "entity_coverage": "", "content_depth": "medium"}
            
            return analysis
            
        except Exception as e:
            # Track Gemini API error
            track_gemini_call("DNA Analysis", prompt, error=e, start_time=start_time)
            
            # Return meaningful fallback instead of empty arrays
            return {
                "relevance_score": 50,
                "key_elements": [
                    {
                        "element_type": "fallback_analysis",
                        "content_snippet": "Content analysis failed - using fallback",
                        "relevance_reason": "API error - manual review needed",
                        "ranking_factor": "unknown"
                    }
                ],
                "content_patterns": [
                    {
                        "pattern": "standard_web_content",
                        "frequency": "medium",
                        "impact": "neutral"
                    }
                ],
                "semantic_density": {
                    "topic_relevance": 50,
                    "entity_coverage": "analysis_failed",
                    "content_depth": "medium"
                },
                "error": str(e)
            }
    
    def _synthesize_dna_profile(self, chunk_analyses: List[Dict[str, Any]], 
                               metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize chunk analyses into comprehensive DNA profile"""
        profile = {
            "page_structure": {
                "html_patterns": [],
                "semantic_markup": [],
                "content_hierarchy": []
            },
            "content_characteristics": {
                "topic_focus": "",
                "entity_density": 0,
                "content_depth": "medium",
                "unique_value_props": []
            },
            "technical_seo": {
                "meta_optimization": {},
                "structured_data": [],
                "performance_signals": []
            },
            "ranking_factors": {
                "evidence_quality": 0,
                "authority_signals": [],
                "user_experience": [],
                "content_freshness": "unknown"
            }
        }
        
        # Aggregate data from chunk analyses
        all_elements = []
        all_patterns = []
        relevance_scores = []
        
        for analysis in chunk_analyses:
            if 'key_elements' in analysis:
                all_elements.extend(analysis['key_elements'])
            if 'content_patterns' in analysis:
                all_patterns.extend(analysis['content_patterns'])
            if 'relevance_score' in analysis:
                relevance_scores.append(analysis['relevance_score'])
        
        # Update profile with aggregated data
        profile["page_structure"]["html_patterns"] = [
            elem.get("element_type", "") for elem in all_elements[:10]
        ]
        
        profile["ranking_factors"]["evidence_quality"] = (
            sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        )
        
        # Add metadata insights
        if metadata:
            profile["technical_seo"]["meta_optimization"] = {
                "title_length": len(metadata.get("title", "")),
                "description_length": len(metadata.get("description", "")),
                "has_canonical": bool(metadata.get("canonical_url")),
                "has_og_tags": bool(metadata.get("og_title"))
            }
        
        return profile
    
    def _extract_causal_evidence(self, chunk_analyses: List[Dict[str, Any]], 
                                chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract causal evidence linking content to AI response"""
        evidence = []
        
        for i, (analysis, chunk) in enumerate(zip(chunk_analyses, chunks)):
            if 'key_elements' in analysis:
                for element in analysis['key_elements'][:3]:  # Top 3 elements per chunk
                    evidence.append({
                        "chunk_id": chunk['chunk_id'],
                        "section_type": chunk['section_type'],
                        "html_snippet": element.get('content_snippet', '')[:200],
                        "relevance_reason": element.get('relevance_reason', ''),
                        "ranking_factor": element.get('ranking_factor', ''),
                        "causal_strength": analysis.get('relevance_score', 0) / 100
                    })
        
        return evidence
    
    def _generate_content_insights(self, dna_profile: Dict[str, Any], 
                                 classification: str) -> Dict[str, Any]:
        """Generate actionable insights from DNA analysis"""
        
        # Extract meaningful data from DNA profile
        evidence_quality = dna_profile.get("ranking_factors", {}).get("evidence_quality", 0)
        entity_coverage = dna_profile.get("content_characteristics", {}).get("topic_focus", "")
        content_depth = dna_profile.get("content_characteristics", {}).get("content_depth", "medium")
        
        # Generate dynamic insights based on actual analysis
        strengths = []
        weaknesses = []
        opportunities = []
        
        if evidence_quality > 70:
            strengths.append("High evidence quality and relevance")
        elif evidence_quality < 30:
            weaknesses.append("Low evidence quality and relevance")
        
        if entity_coverage:
            strengths.append(f"Strong entity coverage: {entity_coverage}")
        else:
            weaknesses.append("Poor entity coverage and topic focus")
        
        if content_depth == "deep":
            strengths.append("Comprehensive content depth")
        elif content_depth == "shallow":
            weaknesses.append("Insufficient content depth")
        
        # Classification-specific insights
        if classification == "third_party":
            strengths.append("Third-party authority perspective")
            opportunities.append("Leverage authority for credibility")
        elif classification == "competitor":
            weaknesses.append("Direct competitor analysis needed")
            opportunities.append("Identify competitive gaps")
        
        insights = {
            "content_strategy": {
                "strengths": strengths if strengths else ["Basic content structure present"],
                "weaknesses": weaknesses if weaknesses else ["Could improve content specificity"],
                "opportunities": opportunities if opportunities else ["Enhance topic coverage"]
            },
            "seo_recommendations": {
                "technical_improvements": [
                    "Optimize meta descriptions for better click-through rates",
                    "Ensure proper heading hierarchy (H1 > H2 > H3)"
                ],
                "content_optimizations": [
                    "Increase entity density and topic coverage",
                    "Add more specific examples and case studies"
                ],
                "structure_enhancements": [
                    "Implement semantic HTML5 elements",
                    "Add structured data markup"
                ]
            },
            "competitive_analysis": {
                "differentiators": [f"{classification.title()} perspective"],
                "gap_opportunities": [
                    "Identify underserved subtopics",
                    "Create comparison content"
                ]
            }
        }
        
        return insights


def analyze_website_dna(classified_data: Dict[str, Any], 
                       ai_response: Dict[str, Any]) -> DNAAnalysisResult:
    """
    Main function to analyze website DNA.
    
    Args:
        classified_data: Output from Stage 1 classification
        ai_response: Original AI response with query and context
        
    Returns:
        DNAAnalysisResult with comprehensive analysis
    """
    analyzer = ForensicDNAAnalyzer()
    return analyzer.analyze_content_dna(classified_data, ai_response)
