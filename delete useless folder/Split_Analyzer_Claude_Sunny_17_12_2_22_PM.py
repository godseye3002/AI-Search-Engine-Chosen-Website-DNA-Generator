import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv() 
API_KEY = os.getenv('GEMINI_API_KEY', '')
genai.configure(api_key=API_KEY)


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
        
        return chunks
    
    def _extract_metadata_chunk(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        head = soup.find('head')
        if not head:
            return None
        
        metadata_html = str(head)
        json_ld = soup.find_all('script', type='application/ld+json')
        print(f"==================== {json_ld}")
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
        
        combined = '\n'.join([p['html'] for p in structural_parts])
        
        return {
            'chunk_id': 'chunk_2_structural',
            'chunk_name': 'Structural/Navigation Layer',
            'raw_html': combined[:self.max_tokens_per_chunk * 4],
            'structured_elements': structural_parts,
            'section_type': 'structural'
        }
    
    def _extract_main_content_with_overlap(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        main = (soup.find('main') or soup.find('article') or 
                soup.find('div', class_=re.compile('content|main', re.I)) or soup.find('body'))
        
        if not main:
            return []
        
        blocks = main.find_all(['p', 'div', 'section', 'h1', 'h2', 'h3', 'ul', 'ol', 'table'], recursive=True)
        
        if not blocks:
            return []
        
        structured = []
        for elem in blocks:
            structured.append({
                'tag': elem.name,
                'html': str(elem),
                'dom_path': self.get_dom_path(elem),
                'has_list': bool(elem.find_all(['ul', 'ol'])),
                'has_table': bool(elem.find_all('table'))
            })
        
        mid = len(structured) // 2
        overlap = max(1, min(3, mid // 10))
        
        top_blocks = structured[:mid + overlap]
        deep_blocks = structured[mid:]
        
        top_html = '\n'.join([b['html'] for b in top_blocks])
        deep_html = '\n'.join([b['html'] for b in deep_blocks])
        
        return [
            {
                'chunk_id': 'chunk_3_main_top',
                'chunk_name': 'Main Content - Top',
                'raw_html': top_html[:self.max_tokens_per_chunk * 4],
                'structured_elements': top_blocks,
                'section_type': 'main_content_top'
            },
            {
                'chunk_id': 'chunk_4_main_deep',
                'chunk_name': 'Main Content - Deep',
                'raw_html': deep_html[:self.max_tokens_per_chunk * 4],
                'structured_elements': deep_blocks,
                'section_type': 'main_content_deep'
            }
        ]
    
    def _extract_sidebar_chunk(self, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
        sidebar_parts = []
        
        for aside in soup.find_all('aside'):
            sidebar_parts.append({'tag': 'aside', 'html': str(aside), 'dom_path': self.get_dom_path(aside)})
        
        for pattern in ['sidebar', 'widget', 'related']:
            for elem in soup.find_all(class_=re.compile(pattern, re.I)):
                sidebar_parts.append({'tag': elem.name, 'html': str(elem)[:1000], 'dom_path': self.get_dom_path(elem)})
        
        if not sidebar_parts:
            return None
        
        combined = '\n'.join([p['html'] for p in sidebar_parts])
        
        return {
            'chunk_id': 'chunk_5_sidebar',
            'chunk_name': 'Sidebar/Footer Layer',
            'raw_html': combined[:self.max_tokens_per_chunk * 4],
            'structured_elements': sidebar_parts,
            'section_type': 'sidebar'
        }


class ForensicDNAAnalyzer:
    """Main analyzer for extracting winning content DNA"""
    
    FORENSIC_ANALYZER_PROMPT = """
### SYSTEM ROLE
You are a Generative Engine Optimization (GEO) Forensic Specialist. Your mission is to reverse-engineer WHY an AI Search Engine cited this webpage.

### CRITICAL CONTEXT
This analysis will feed into a "Reconstruction AI" that builds high-ranking content. You must extract the COMPLETE DNA of winning content, not summarize it.

### GEO PRINCIPLES TO INVESTIGATE
Based on research (Ahrefs, Google SGE studies), look for these citation factors:

1. **Information Gain**: Unique statistics, data tables, fresh numbers
2. **Citation & Quotations**: Direct expert quotes, external study citations
3. **Fluency & Structure**: Clear lists (ul/ol), tables, simple SVO sentences
4. **Semantic Proximity**: How close answers are to entities in the DOM (e.g., price next to product name)

### THE GOLDEN THREAD
You will receive:
- `AI_OVERVIEW_RESPONSE`: The exact text the AI search engine outputted
- `SOURCE_HTML_CHUNK`: A portion of the cited webpage with DOM structure preserved

### YOUR TASK
Scan this HTML chunk forensically. For EVERY piece of content that appears to correlate with the AI response:

1. Extract the exact HTML snippet (with tags)
2. Generate the DOM signature (e.g., "DIV > UL > LI > STRONG")
3. Hypothesize WHY this specific element was valuable
4. Map it to the corresponding sentence in the AI response

### SPECIAL INSTRUCTION FOR STRUCTURED DATA (JSON-LD)
If you find "application/ld+json", analyze it differently:
1. Does it establish AUTHORITATIVENESS (Author, Publisher)? 
2. Does it define the ENTITY (Organization, Service, Product)?
3. Even if the text doesn't match the AI response word-for-word, did this Schema likely help the AI understand the *context* of the page?
   - If YES -> Mark as 'Schema_Authority_Signal' with a hypothesis on how it helped indexing/entity resolution.

### OUTPUT FORMAT
Output ONLY valid JSON. No markdown, no commentary.

### JSON SCHEMA
{
  "chunk_id": "string",
  "evidence_density_score": float (0.0 to 1.0, how much of this chunk matches the AI response),
  "extracted_signals": [
    {
      "signal_type": "string (options: 'Direct_Text_Match', 'Structural_Mirror', 'Unique_Data_Point', 'Entity_Co_occurrence', 'Expert_Citation', 'List_Format', 'Table_Data', 'Semantic_Proximity')",
      "hypothesis": "string (explain WHY the AI likely used this element)",
      "exact_html_snippet": "string (the raw HTML from the chunk, preserving tags)",
      "dom_signature": "string (e.g., 'DIV.content > UL > LI > STRONG')",
      "ai_response_correlation": "string (the specific phrase/sentence in the AI response that matches)",
      "information_gain_score": float (0-1, uniqueness of data),
      "structural_advantage": "string (e.g., 'Bulleted list format', 'Table with clear headers')",
      "schema_authority_signal": "string (if this is a schema authority signal, describe it"
    }
  ],
  "content_dna_local": {
    "readability_level": "string (Flesch-Kincaid equivalent: 'Elementary', 'High School', 'College')",
    "formatting_style": "string (e.g., 'Data-heavy table', 'Narrative with inline stats', 'FAQ format')",
    "dominant_tags": ["list", "of", "most", "common", "HTML", "tags"],
    "has_expert_quotes": boolean,
    "has_external_citations": boolean
  }
}

### CRITICAL INSTRUCTIONS
- If you find a match, include the ENTIRE HTML element, not just the text
- Preserve all attributes (class, id, data-*)
- If nothing matches, return evidence_density_score: 0.0 and empty extracted_signals
- Look for subtle matches: paraphrases, synonym usage, data reformatting
"""

    DNA_SYNTHESIZER_PROMPT = """
### SYSTEM ROLE
You are a Content DNA Archivist. Your job is to aggregate ALL evidence from multiple HTML chunks into a Master Blueprint.

### CRITICAL CONTEXT
This output will be compared against 50+ other winning pages. Your job is NOT to pick the "best" chunk, but to preserve EVERY valid signal so the downstream Reconstruction AI can identify patterns across all winning content.

### THE GOLDEN THREAD
You will receive:
- The original `AI_OVERVIEW_RESPONSE` (what the AI search engine said)
- Analysis results from up to 5 HTML chunks (Metadata, Structural, Main Top, Main Deep, Sidebar)

### YOUR TASK
1. Aggregate ALL `extracted_signals` from all chunks into one master list
2. Identify the page's overall structure type (Listicle, Comparison, Guide, FAQ, etc.)
3. Determine dominant HTML patterns (lists, tables, emphasis tags)
4. Perform gap analysis: what did the AI mention that this page DIDN'T have?

### OUTPUT FORMAT
Output ONLY valid JSON.

### JSON SCHEMA
{
  "citation_url": "string",
  "ai_overview_context": {
    "query": "string",
    "response_fragment": "string (first 500 chars of AI response)"
  },
  "winning_dna_profile": {
    "page_structure_type": "string (options: 'Listicle', 'Comparison_Review', 'Definition_Guide', 'FAQ', 'How_To_Tutorial', 'Data_Report', 'Mixed')",
    "dominant_html_patterns": ["ordered list of most effective DOM signatures found"],
    "semantic_density": "string (High/Medium/Low - how much unique info per paragraph)",
    "citation_factors_present": ["Information_Gain", "Expert_Citations", "Structural_Clarity", "Semantic_Proximity"],
    "estimated_word_count": integer,
    "estimated_readability": "string"
  },
  "causal_evidence_chain": [
    {
      "chunk_source": "string (which chunk this came from)",
      "signal_type": "string",
      "hypothesis": "string",
      "exact_html_snippet": "string (preserve the HTML)",
      "dom_signature": "string",
      "ai_response_match": "string",
      "information_gain_score": float,
      "structural_advantage": "string"
    }
  ],
  "missing_opportunities": [
    "string (things mentioned in AI response but not found in this source)"
  ],
  "synthesis_summary": "string (2-3 sentences explaining the overall DNA of this winning page)"
}

### AGGREGATION RULES
1. Preserve ALL signals with evidence_density_score > 0.3
2. Sort causal_evidence_chain by information_gain_score (descending)
3. If multiple chunks found the same signal, keep the one with better structural_advantage
4. For missing_opportunities, identify high-value keywords or entities present in the AI_OVERVIEW_RESPONSE that were NOT detected in the provided extracted_signals. Only list items that are demonstrably absent from the analyzed chunks.
"""
    
    def __init__(self, api_key: str = None):
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.chunker = SemanticHTMLChunker()
    
    def analyze(self, classified_data: Dict, metadata: Dict, ai_response: Dict) -> Dict:
        """Main forensic analysis pipeline - The Golden Thread flows through all steps"""
        print("="*60)
        print("🔬 FORENSIC DNA ANALYSIS PIPELINE")
        print("="*60)
        
        html_content = classified_data.get('html', '')
        url = classified_data.get('url', 'unknown')
        
        # Extract AI Overview with Golden Thread
        ai_overview_json = {
            'query': ai_response.get('query', ''),
            'response_text': ai_response.get('ai_overview_text', '') or str(ai_response),
            'full_context': ai_response
        }
        
        print(f"\n📋 Query: {ai_overview_json['query']}")
        print(f"📋 AI Response Length: {len(ai_overview_json['response_text'])} chars")
        print(f"📋 Source URL: {url}\n")
        
        # Phase 1: Smart Semantic Chunking (max 5, with overlap)
        print("Phase 1: Smart Semantic Chunking...")
        chunks = self.chunker.chunk_html_with_structure(html_content)
        print(f"✅ Created {len(chunks)} semantic chunks with DOM structure preserved")
        for chunk in chunks:
            print(f"   - {chunk['chunk_name']} ({chunk['section_type']})")
        
        # Phase 2: Forensic Analysis (Evidence Collection)
        print("\nPhase 2: Forensic Evidence Collection...")
        chunk_analyses = []
        for idx, chunk in enumerate(chunks):
            print(f"   🔍 Analyzing {chunk['chunk_name']}...")
            analysis = self._forensic_chunk_analysis(
                chunk=chunk,
                ai_overview_json=ai_overview_json,
                metadata=metadata
            )
            if analysis:
                density = analysis.get('evidence_density_score', 0)
                signals = len(analysis.get('extracted_signals', []))
                print(f"      ✓ Density: {density:.2f}, Signals: {signals}")
                chunk_analyses.append(analysis)
        
        # Phase 3: DNA Synthesis (Archival Aggregation)
        print("\nPhase 3: DNA Synthesis & Archival...")
        final_dna = self._synthesize_winning_dna(
            chunk_analyses=chunk_analyses,
            ai_overview_json=ai_overview_json,
            url=url
        )
        
        print(f"\n✅ DNA EXTRACTION COMPLETE")
        print(f"   Total Evidence Signals: {len(final_dna.get('causal_evidence_chain', []))}")
        print(f"   Page Structure Type: {final_dna.get('winning_dna_profile', {}).get('page_structure_type', 'Unknown')}")
        print(f"   Missing Opportunities: {len(final_dna.get('missing_opportunities', []))}")
        
        return final_dna
    
    def _forensic_chunk_analysis(self, chunk: Dict, ai_overview_json: Dict, metadata: Dict) -> Dict:
        """Phase 2: Forensic evidence extraction with Golden Thread"""
        
        prompt = f"""
{self.FORENSIC_ANALYZER_PROMPT}

### INPUT DATA

**AI Overview Response (The Golden Thread):**
Query: {ai_overview_json['query']}
Response Text:
{ai_overview_json['response_text'][:2500]}

**Source HTML Chunk:**
Chunk Name: {chunk['chunk_name']}
Section Type: {chunk['section_type']}

```html
{chunk['raw_html'][:10000]}
```

**Page Metadata Context:**
- Title: {metadata.get('title', '')}
- Description: {metadata.get('description', '')}
- Author: {metadata.get('author', '')}

### YOUR FORENSIC TASK
Scan this chunk for ANY correlation with the AI response. Extract exact HTML snippets with DOM signatures. Output ONLY JSON.
"""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            result_text = re.sub(r'^```json\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
            
            result = json.loads(result_text)
            result['chunk_id'] = chunk['chunk_id']
            result['chunk_name'] = chunk['chunk_name']
            result['section_type'] = chunk['section_type']
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"      ⚠️ JSON decode error: {e}")
            return self._create_empty_analysis(chunk)
        except Exception as e:
            print(f"      ⚠️ Analysis error: {e}")
            return self._create_empty_analysis(chunk)
    
    def _synthesize_winning_dna(self, chunk_analyses: List[Dict], ai_overview_json: Dict, url: str) -> Dict:
        """Phase 3: Aggregate all evidence into Master DNA Blueprint"""
        
        if not chunk_analyses:
            return self._create_empty_dna_profile(url, ai_overview_json)
        
        # Filter for meaningful evidence
        valid_analyses = [a for a in chunk_analyses if a.get('evidence_density_score', 0) > 0.1]
        
        if not valid_analyses:
            print("   ⚠️ No significant evidence found, using fallback")
            return self._fallback_dna_synthesis(chunk_analyses, url, ai_overview_json)
        
        # Prepare synthesis input
        synthesis_data = {
            "ai_overview": ai_overview_json,
            "chunk_analyses": valid_analyses
        }
        
        prompt = f"""
{self.DNA_SYNTHESIZER_PROMPT}

### INPUT DATA

**AI Overview Response (The Golden Thread):**
Query: {ai_overview_json['query']}
Response:
{ai_overview_json['response_text'][:2000]}

**Chunk Analysis Results:**
```json
{json.dumps(valid_analyses, indent=2)[:15000]}
```

### YOUR ARCHIVAL TASK
Aggregate ALL evidence into a Master DNA Blueprint. Preserve every signal, don't discard. Perform gap analysis. Output ONLY JSON.
"""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            result_text = re.sub(r'^```json\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
            
            dna_profile = json.loads(result_text)
            dna_profile['citation_url'] = url
            
            # Ensure all fields exist
            if 'ai_overview_context' not in dna_profile:
                dna_profile['ai_overview_context'] = {
                    'query': ai_overview_json['query'],
                    'response_fragment': ai_overview_json['response_text'][:500]
                }
            
            return dna_profile
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️ Synthesis JSON error: {e}, using fallback")
            return self._fallback_dna_synthesis(valid_analyses, url, ai_overview_json)
        except Exception as e:
            print(f"   ⚠️ Synthesis error: {e}, using fallback")
            return self._fallback_dna_synthesis(valid_analyses, url, ai_overview_json)
    
    def _fallback_dna_synthesis(self, analyses: List[Dict], url: str, ai_overview: Dict) -> Dict:
        """Fallback DNA aggregation if LLM synthesis fails"""
        
        all_signals = []
        all_patterns = []
        citation_factors = set()
        
        for analysis in analyses:
            signals = analysis.get('extracted_signals', [])
            for signal in signals:
                all_signals.append({
                    'chunk_source': analysis.get('chunk_name', 'unknown'),
                    'signal_type': signal.get('signal_type', 'Unknown'),
                    'hypothesis': signal.get('hypothesis', ''),
                    'exact_html_snippet': signal.get('exact_html_snippet', '')[:500],
                    'dom_signature': signal.get('dom_signature', ''),
                    'ai_response_match': signal.get('ai_response_correlation', ''),
                    'information_gain_score': signal.get('information_gain_score', 0.5),
                    'structural_advantage': signal.get('structural_advantage', '')
                })
            
            dna = analysis.get('content_dna_local', {})
            if dna.get('has_expert_quotes'):
                citation_factors.add('Expert_Citations')
            if dna.get('has_external_citations'):
                citation_factors.add('External_Citations')
            
            dom_tags = dna.get('dominant_tags', [])
            all_patterns.extend(dom_tags)
        
        # Sort by information gain
        all_signals.sort(key=lambda x: x.get('information_gain_score', 0), reverse=True)
        
        # Determine page type from patterns
        page_type = 'Mixed'
        if 'ul' in all_patterns or 'ol' in all_patterns:
            page_type = 'Listicle'
        if 'table' in all_patterns:
            page_type = 'Comparison_Review'
        
        return {
            "citation_url": url,
            "ai_overview_context": {
                "query": ai_overview['query'],
                "response_fragment": ai_overview['response_text'][:500]
            },
            "winning_dna_profile": {
                "page_structure_type": page_type,
                "dominant_html_patterns": list(set(all_patterns))[:10],
                "semantic_density": "Medium",
                "citation_factors_present": list(citation_factors),
                "estimated_word_count": sum([len(a.get('raw_html', '').split()) for a in analyses]),
                "estimated_readability": "Unknown"
            },
            "causal_evidence_chain": all_signals,
            "missing_opportunities": [],
            "synthesis_summary": f"Fallback synthesis aggregated {len(all_signals)} signals from {len(analyses)} chunks."
        }
    
    def _create_empty_analysis(self, chunk: Dict) -> Dict:
        """Empty analysis for failed chunks"""
        return {
            'chunk_id': chunk['chunk_id'],
            'chunk_name': chunk['chunk_name'],
            'section_type': chunk['section_type'],
            'evidence_density_score': 0.0,
            'extracted_signals': [],
            'content_dna_local': {
                'readability_level': 'Unknown',
                'formatting_style': 'Unknown',
                'dominant_tags': [],
                'has_expert_quotes': False,
                'has_external_citations': False
            }
        }
    
    def _create_empty_dna_profile(self, url: str, ai_overview: Dict) -> Dict:
        """Empty DNA profile when no evidence found"""
        return {
            "citation_url": url,
            "ai_overview_context": {
                "query": ai_overview['query'],
                "response_fragment": ai_overview['response_text'][:500]
            },
            "winning_dna_profile": {
                "page_structure_type": "Unknown",
                "dominant_html_patterns": [],
                "semantic_density": "Low",
                "citation_factors_present": [],
                "estimated_word_count": 0,
                "estimated_readability": "Unknown"
            },
            "causal_evidence_chain": [],
            "missing_opportunities": [],
            "synthesis_summary": "No significant evidence found in any chunk."
        }


def main():
    """Main execution function"""
    
    print("Loading input data...")
    
    with open('classified_data.json', 'r', encoding='utf-8') as f:
        classified_data = json.load(f)
    
    with open('metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    with open('ai_response.json', 'r', encoding='utf-8') as f:
        ai_response = json.load(f)
    
    # Initialize forensic analyzer
    analyzer = ForensicDNAAnalyzer()
    
    # Run DNA extraction
    dna_profile = analyzer.analyze(
        classified_data=classified_data,
        metadata=metadata,
        ai_response=ai_response
    )
    
    # Save DNA blueprint
    output_file = 'winning_content_dna2.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dna_profile, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"💾 DNA BLUEPRINT SAVED: {output_file}")
    print(f"{'='*60}")
    print(f"\n📊 FINAL DNA PROFILE:")
    print(f"   URL: {dna_profile['citation_url']}")
    print(f"   Structure Type: {dna_profile['winning_dna_profile']['page_structure_type']}")
    print(f"   Evidence Signals: {len(dna_profile['causal_evidence_chain'])}")
    print(f"   DOM Patterns: {len(dna_profile['winning_dna_profile']['dominant_html_patterns'])}")
    print(f"   Missing Opportunities: {len(dna_profile['missing_opportunities'])}")
    
    if dna_profile['causal_evidence_chain']:
        print(f"\n🔝 TOP SIGNALS:")
        for signal in dna_profile['causal_evidence_chain'][:3]:
            print(f"   - {signal['signal_type']}: {signal['hypothesis'][:80]}...")
    
    print(f"\n📝 Synthesis: {dna_profile['synthesis_summary']}")
    
    return dna_profile


if __name__ == "__main__":
    main()