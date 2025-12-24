"""
Final Aggregation Core - Enhanced with Master Blueprint Logic

Aggregates DNA analysis results from Stage 2 into comprehensive master blueprints.
Combines the sophisticated Master Blueprint approach with competitive insights.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() 
API_KEY = os.getenv('GEMINI_API_KEY', '')
genai.configure(api_key=API_KEY)


@dataclass
class AggregationResult:
    """Result of final aggregation for a pipeline run"""
    run_id: str
    query: str
    total_analyzed: int
    master_blueprint: Dict[str, Any]
    competitive_insights: Dict[str, Any]
    content_recommendations: List[Dict[str, Any]]
    ranking_opportunities: List[Dict[str, Any]]
    summary_report: str
    error: Optional[str] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'run_id': self.run_id,
            'query': self.query,
            'total_analyzed': self.total_analyzed,
            'master_blueprint': self.master_blueprint,
            'competitive_insights': self.competitive_insights,
            'content_recommendations': self.content_recommendations,
            'ranking_opportunities': self.ranking_opportunities,
            'summary_report': self.summary_report,
            'error': self.error,
            'processing_time': self.processing_time,
            'aggregation_timestamp': datetime.now().isoformat()
        }


class MasterBlueprintAggregator:
    """Enhanced aggregator using Master Blueprint methodology"""
    
    MASTER_BLUEPRINT_PROMPT = """
**Role:** Master Information Architect & DNA Synthesis Engine
**Task:** Aggregate multiple "DNA Analysis Results" (JSON) into a single, high-efficiency "Master Blueprint."
**Goal:** Create a unified data structure that holds *maximum* amount of unique, high-value information with *minimum* amount of redundancy, optimized for content strategy and SEO.

---

### **Context & Objective**
You are an advanced DNA synthesizer. Your job is to ingest N number of DNA analysis JSON files from website analysis. Each file represents a high-ranking webpage's DNA profile. Your goal is to combine them into one **"Master Blueprint JSON"**.

This Master Blueprint must not be a simple concatenation. It must be a **smart reconstruction** that follows these core engineering principles:
1.  **Structural Consensus:** Identify most common HTML layout patterns (e.g., H1 followed by P, then H2) and establish them as `core_layout_schema`.
2.  **Entity Deduplication:** If "Entity_Alpha" appears in 10 files, store it *once* as a primary entity.
3.  **Information Gain Maximization (The Golden Rule):**
    * **Consensus Data (Low Entropy):** If 8/10 sources describe an entity generically (e.g., "Standard Industry Solution"), store this as `primary_definition`.
    * **Unique Data (High Entropy):** If only 1/10 sources provides a specific statistic (e.g., "Value_X" or "Percentage_Y"), you **MUST** preserve this as a `high_value_attribute`. Do not discard rare data if it contains numbers, percentages, or proper nouns.
4.  **Noise Filtration:** Discard generic fluff (e.g., "Best in class," "Amazing performance") that carries no factual weight.

---

### **The Logic: How to Process Input**

**Phase 1: Structural Analysis (The Skeleton)**
Scan `page_structure` of all inputs. Determine "Winning Sequence."
* *Logic:* If majority of successful pages use a "Comparison Module" after "Introduction," your Master Blueprint must enforce a `required_module: comparison_module` at that position.

**Phase 2: Entity Fusion (The Data)**
Create a centralized database of all entities (products, concepts, people) found across inputs.
* **Merge Strategy:**
    * *Input A:* "Entity_Alpha is a Type_1 tool."
    * *Input B:* "Entity_Alpha solves Problem_X."
    * *Input C:* "Entity_Alpha has Feature_Z."
    * *Result:* Store `Entity: Entity_Alpha` with `attributes: ["Type_1 tool", "Solves Problem_X", "Feature_Z"]`.

**Phase 3: Validation Layers (The Quality Control)**
Identify mandatory elements required for content to be considered "Complete" by a search engine.
* *Logic:* If high-ranking pages always cite "Regulatory Standard A" when discussing "Topic B," create a rule: `dependency: Topic B requires Regulatory_Standard_A`.

---

### **Target Output Format (JSON Schema)**

You must output final result in following JSON structure. **Do not use biological terms.** Use keys exactly as defined below:

```json
{
  "master_blueprint_id": "dna_aggregated_v1",
  "blueprint_generation_logic": "Consensus_Plus_Uniqueness",
  "query_context": "{query}",
  "sources_analyzed": {total_sources},
  
  "layer_1_structural_framework": [
    {
      "sequence_order": 1,
      "component_type": "h1_header",
      "consensus_intent": "Define Topic clearly",
      "optimized_pattern_template": "The [Adjective] Guide to [Topic] for [Audience]",
      "adoption_rate": 0.8
    },
    {
      "sequence_order": 2,
      "component_type": "summary_block",
      "content_requirement": "Must include key entities: [List_Top_Entities_Found]",
      "adoption_rate": 0.7
    }
  ],

  "layer_2_entity_knowledge_base": {
    "Entity_Name_Here": {
      "classification": "Primary Category / Tool Type",
      "primary_definition": "The most common description found across sources.",
      "high_value_attributes": [
        {
          "data_point": "Specific statistic or unique fact",
          "source_weight": "High (derived from unique signal)",
          "confidence": 0.9
        }
      ],
      "semantic_keywords": ["List", "Related", "Keywords"],
      "frequency_across_sources": 0.6
    }
  },

  "layer_3_content_patterns": {
    "successful_formats": [
      {
        "format_type": "comparison_table",
        "usage_frequency": 0.8,
        "effectiveness_score": 0.9
      }
    ],
    "content_depth_indicators": {
      "average_word_count": 1500,
      "entity_density_optimal": 0.05,
      "readability_score": 0.7
    }
  },

  "layer_4_seo_compliance": {
    "mandatory_inclusions": [
      "List of items that MUST be present to avoid low-quality scoring"
    ],
    "negative_constraints": [
      "List of patterns to avoid"
    ],
    "technical_requirements": {
      "page_load_speed_optimal": "< 2s",
      "mobile_friendly": true,
      "structured_data_present": true
    }
  }
}
```

---

**CRITICAL INSTRUCTIONS:**
1. Analyze ALL provided DNA analysis JSON files
2. Generate output in STRICT JSON format only
3. No markdown, no explanations outside JSON
4. Follow exact schema provided above
5. Preserve all unique high-value data points
6. Include adoption rates and effectiveness scores based on frequency across sources
"""
    
    def __init__(self, model_name: str = "gemini-3-pro-preview"):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
    
    def aggregate_run_results(self, run_id: str, query: str, 
                            dna_results: List[Dict[str, Any]]) -> AggregationResult:
        """
        Aggregate DNA analysis results for a complete pipeline run.
        
        Args:
            run_id: Pipeline run identifier
            query: Original search query
            dna_results: List of DNA analysis results from Stage 2
            
        Returns:
            AggregationResult with comprehensive insights
        """
        start_time = datetime.now()
        
        try:
            # Filter valid results
            valid_results = [result for result in dna_results 
                           if result.get('error') is None and result.get('dna_profile')]
            
            if not valid_results:
                raise ValueError("No valid DNA analysis results to aggregate")
            
            # Aggregate DNA profiles
            aggregated_profile = self._aggregate_dna_profiles(valid_results)
            
            # Generate master blueprint
            master_blueprint = self._generate_master_blueprint(dna_results, query)
            
            # Generate competitive insights
            competitive_insights = self._generate_competitive_insights(valid_results, query)
            
            # Create content recommendations
            content_recommendations = self._generate_content_recommendations(master_blueprint, competitive_insights, query)
            
            # Identify ranking opportunities
            ranking_opportunities = self._identify_ranking_opportunities(valid_results, master_blueprint)
            
            # Generate summary report
            summary_report = self._generate_summary_report(query, valid_results, master_blueprint, 
                                                          competitive_insights, content_recommendations)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AggregationResult(
                run_id=run_id,
                query=query,
                total_analyzed=len(valid_results),
                master_blueprint=master_blueprint,
                competitive_insights=competitive_insights,
                content_recommendations=content_recommendations,
                ranking_opportunities=ranking_opportunities,
                summary_report=summary_report,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return AggregationResult(
                run_id=run_id,
                query=query,
                total_analyzed=0,
                master_blueprint={},
                competitive_insights={},
                content_recommendations=[],
                ranking_opportunities=[],
                summary_report="",
                error=str(e),
                processing_time=processing_time
            )
    
    def _aggregate_dna_profiles(self, dna_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual DNA profiles into comprehensive profile"""
        aggregated = {
            "page_structure": {
                "common_html_patterns": {},
                "semantic_markup_frequency": {},
                "content_hierarchy_patterns": []
            },
            "content_characteristics": {
                "topic_focus_areas": [],
                "average_entity_density": 0,
                "content_depth_distribution": {},
                "common_value_props": []
            },
            "technical_seo": {
                "meta_optimization_patterns": {},
                "structured_data_usage": {},
                "performance_signal_patterns": []
            },
            "ranking_factors": {
                "average_evidence_quality": 0,
                "common_authority_signals": [],
                "user_experience_patterns": [],
                "content_freshness_indicators": {}
            }
        }
        
        # Aggregate data from all results
        all_evidence_scores = []
        all_entity_densities = []
        html_pattern_counts = {}
        
        for result in dna_results:
            dna_profile = result.get('dna_profile', {})
            
            # Collect evidence quality scores
            ranking_factors = dna_profile.get('ranking_factors', {})
            if 'evidence_quality' in ranking_factors:
                all_evidence_scores.append(ranking_factors['evidence_quality'])
            
            # Collect entity densities
            content_chars = dna_profile.get('content_characteristics', {})
            if 'entity_density' in content_chars:
                all_entity_densities.append(content_chars['entity_density'])
            
            # Collect HTML patterns
            page_structure = dna_profile.get('page_structure', {})
            html_patterns = page_structure.get('html_patterns', [])
            for pattern in html_patterns:
                html_pattern_counts[pattern] = html_pattern_counts.get(pattern, 0) + 1
        
        # Calculate aggregated metrics
        if all_evidence_scores:
            aggregated["ranking_factors"]["average_evidence_quality"] = (
                sum(all_evidence_scores) / len(all_evidence_scores)
            )
        
        if all_entity_densities:
            aggregated["content_characteristics"]["average_entity_density"] = (
                sum(all_entity_densities) / len(all_entity_densities)
            )
        
        # Get most common HTML patterns
        if html_pattern_counts:
            sorted_patterns = sorted(html_pattern_counts.items(), 
                                  key=lambda x: x[1], reverse=True)
            aggregated["page_structure"]["common_html_patterns"] = {
                pattern: count for pattern, count in sorted_patterns[:10]
            }
        
        return aggregated
    
    def _generate_master_blueprint(self, dna_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Generate master blueprint using enhanced prompt"""
        try:
            # Prepare input data for prompt
            dna_profiles = [result.get('dna_profile', {}) for result in dna_results]
            
            input_data = {
                "query": query,
                "total_sources": len(dna_results),
                "dna_profiles": dna_profiles
            }
            
            # Create full prompt
            full_prompt = f"{self.MASTER_BLUEPRINT_PROMPT}\n\n**INPUT DATA:**\n```json\n{json.dumps(input_data, indent=2)}\n```"
            
            # Generate master blueprint using Gemini
            response = self.model.generate_content(full_prompt)
            
            # Extract and parse JSON from response
            response_text = response.text.strip()
            
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()
            
            # Parse JSON
            master_blueprint = json.loads(response_text)
            
            # Add metadata
            master_blueprint['generation_metadata'] = {
                'sources_count': len(dna_results),
                'query': query,
                'generation_timestamp': datetime.now().isoformat()
            }
            
            return master_blueprint
            
        except json.JSONDecodeError as e:
            # Fallback to basic aggregation if JSON parsing fails
            return self._fallback_aggregation(dna_results, query, str(e))
        except Exception as e:
            raise Exception(f"Failed to generate master blueprint: {str(e)}")
    
    def _fallback_aggregation(self, dna_results: List[Dict[str, Any]], query: str, error: str) -> Dict[str, Any]:
        """Fallback aggregation if advanced blueprint generation fails"""
        return {
            "master_blueprint_id": "fallback_v1",
            "blueprint_generation_logic": "Basic_Aggregation",
            "query_context": query,
            "sources_analyzed": len(dna_results),
            "generation_error": error,
            "layer_1_structural_framework": [
                {
                    "sequence_order": 1,
                    "component_type": "h1_header",
                    "consensus_intent": "Define Topic clearly"
                }
            ],
            "layer_2_entity_knowledge_base": {},
            "layer_3_content_patterns": {},
            "layer_4_seo_compliance": {}
        }
    
    def _generate_competitive_insights(self, dna_results: List[Dict[str, Any]], 
                                     query: str) -> Dict[str, Any]:
        """Generate competitive landscape insights"""
        insights = {
            "competitive_landscape": {
                "total_competitors_analyzed": 0,
                "third_party_sources": 0,
                "competitor_sources": 0,
                "authority_distribution": {}
            },
            "content_gaps": [],
            "opportunity_areas": [],
            "market_positioning": {}
        }
        
        # Count source types
        for result in dna_results:
            classification = result.get('classification', '')
            if classification == 'third_party':
                insights["competitive_landscape"]["third_party_sources"] += 1
            elif classification == 'competitor':
                insights["competitive_landscape"]["competitor_sources"] += 1
        
        insights["competitive_landscape"]["total_competitors_analyzed"] = len(dna_results)
        
        # Generate AI-powered insights
        prompt = f"""
        Analyze the competitive landscape based on DNA analysis results:
        
        Query: {query}
        Number of sources analyzed: {len(dna_results)}
        Third-party sources: {insights['competitive_landscape']['third_party_sources']}
        Competitor sources: {insights['competitive_landscape']['competitor_sources']}
        
        DNA Analysis Summary:
        {json.dumps([result.get('dna_profile', {}) for result in dna_results[:3]], indent=2)[:2000]}
        
        Provide competitive insights in JSON format:
        {{
            "market_positioning": {{
                "market_maturity": "emerging/mature/saturated",
                "competition_level": "low/medium/high",
                "content_quality_bar": "description"
            }},
            "content_gaps": [
                {{
                    "gap_type": "topic/format/technical",
                    "description": "what's missing",
                    "opportunity_level": "low/medium/high"
                }}
            ],
            "opportunity_areas": [
                {{
                    "area": "specific opportunity",
                    "rationale": "why it's an opportunity",
                    "difficulty": "easy/medium/hard"
                }}
            ]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            ai_insights = json.loads(response.text)
            
            # Merge AI insights with our analysis
            insights.update(ai_insights)
            
        except Exception as e:
            # Fallback to basic insights
            insights["market_positioning"] = {
                "market_maturity": "unknown",
                "competition_level": "medium",
                "content_quality_bar": "moderate"
            }
        
        return insights
    
    def _generate_content_recommendations(self, aggregated_profile: Dict[str, Any],
                                         competitive_insights: Dict[str, Any],
                                         query: str) -> List[Dict[str, Any]]:
        """Generate actionable content recommendations"""
        recommendations = []
        
        # Technical SEO recommendations
        tech_seo = aggregated_profile.get('technical_seo', {})
        meta_patterns = tech_seo.get('meta_optimization_patterns', {})
        
        if meta_patterns:
            avg_title_length = meta_patterns.get('title_length', 0)
            if avg_title_length < 50:
                recommendations.append({
                    "category": "technical_seo",
                    "priority": "high",
                    "recommendation": "Optimize title tags for better length (50-60 characters)",
                    "implementation": "Review and rewrite title tags to be more descriptive",
                    "expected_impact": "Improved CTR and ranking potential"
                })
        
        # Content structure recommendations
        page_structure = aggregated_profile.get('page_structure', {})
        common_patterns = page_structure.get('common_html_patterns', {})
        
        if 'H1' not in common_patterns:
            recommendations.append({
                "category": "content_structure",
                "priority": "high",
                "recommendation": "Implement proper H1 hierarchy",
                "implementation": "Add descriptive H1 tags to all pages",
                "expected_impact": "Better content organization and SEO"
            })
        
        # Content depth recommendations
        evidence_quality = aggregated_profile.get('ranking_factors', {}).get('average_evidence_quality', 0)
        
        if evidence_quality < 60:
            recommendations.append({
                "category": "content_quality",
                "priority": "high",
                "recommendation": "Increase content depth and evidence quality",
                "implementation": "Add more supporting data, examples, and citations",
                "expected_impact": "Higher relevance and authority signals"
            })
        
        # AI-powered recommendations
        prompt = f"""
        Generate content recommendations based on analysis:
        
        Query: {query}
        Average Evidence Quality: {evidence_quality}
        Common HTML Patterns: {list(common_patterns.keys())[:5]}
        
        Competitive Insights: {json.dumps(competitive_insights, indent=2)[:1000]}
        
        Provide 3-5 specific recommendations in JSON format:
        [
            {{
                "category": "content/technical/strategic",
                "priority": "high/medium/low",
                "recommendation": "specific action",
                "implementation": "how to implement",
                "expected_impact": "expected benefit"
            }}
        ]
        """
        
        try:
            response = self.model.generate_content(prompt)
            ai_recommendations = json.loads(response.text)
            recommendations.extend(ai_recommendations)
        except Exception:
            # Use existing recommendations if AI fails
            pass
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _identify_ranking_opportunities(self, dna_results: List[Dict[str, Any]],
                                      aggregated_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific ranking improvement opportunities"""
        opportunities = []
        
        # Analyze causal evidence patterns
        all_evidence = []
        for result in dna_results:
            evidence = result.get('causal_evidence', [])
            all_evidence.extend(evidence)
        
        # Group evidence by ranking factors
        factor_counts = {}
        for evidence_item in all_evidence:
            factor = evidence_item.get('ranking_factor', 'unknown')
            if factor not in factor_counts:
                factor_counts[factor] = []
            factor_counts[factor].append(evidence_item)
        
        # Identify high-impact opportunities
        for factor, evidence_list in factor_counts.items():
            if len(evidence_list) >= 2:  # Factor appears in multiple sources
                avg_strength = sum(item.get('causal_strength', 0) for item in evidence_list) / len(evidence_list)
                
                if avg_strength > 0.7:
                    opportunities.append({
                        "ranking_factor": factor,
                        "opportunity_type": "high_impact_factor",
                        "evidence_count": len(evidence_list),
                        "average_strength": avg_strength,
                        "recommendation": f"Optimize for {factor} based on successful competitor patterns",
                        "priority": "high" if avg_strength > 0.8 else "medium"
                    })
        
        # Content gap opportunities
        content_chars = aggregated_profile.get('content_characteristics', {})
        entity_density = content_chars.get('average_entity_density', 0)
        
        if entity_density < 2.0:
            opportunities.append({
                "ranking_factor": "entity_coverage",
                "opportunity_type": "content_enrichment",
                "evidence_count": 0,
                "average_strength": 0.6,
                "recommendation": "Increase entity density and topic coverage",
                "priority": "medium"
            })
        
        return opportunities[:8]  # Limit to top 8 opportunities
    
    def _generate_summary_report(self, query: str, dna_results: List[Dict[str, Any]],
                                aggregated_profile: Dict[str, Any],
                                competitive_insights: Dict[str, Any],
                                recommendations: List[Dict[str, Any]]) -> str:
        """Generate comprehensive summary report"""
        
        # Calculate key metrics
        total_sources = len(dna_results)
        avg_evidence_quality = aggregated_profile.get('ranking_factors', {}).get('average_evidence_quality', 0)
        competitor_count = competitive_insights.get('competitive_landscape', {}).get('competitor_sources', 0)
        third_party_count = competitive_insights.get('competitive_landscape', {}).get('third_party_sources', 0)
        
        high_priority_recs = [rec for rec in recommendations if rec.get('priority') == 'high']
        
        prompt = f"""
        Generate a comprehensive summary report for content strategy based on DNA analysis:
        
        ANALYSIS SUMMARY:
        - Query: {query}
        - Sources Analyzed: {total_sources}
        - Average Evidence Quality: {avg_evidence_quality:.1f}%
        - Competitor Sources: {competitor_count}
        - Third-party Sources: {third_party_count}
        - High Priority Recommendations: {len(high_priority_recs)}
        
        KEY INSIGHTS:
        - Competitive Landscape: {json.dumps(competitive_insights.get('market_positioning', {}), indent=2)}
        - Top Recommendations: {json.dumps(high_priority_recs[:3], indent=2)}
        
        Generate a professional summary report with:
        1. Executive Summary
        2. Key Findings
        3. Competitive Analysis
        4. Strategic Recommendations
        5. Next Steps
        
        Format as a clear, actionable business report.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Fallback summary
            return f"""
            DNA Analysis Summary Report
            
            Executive Summary:
            Analyzed {total_sources} sources for query: "{query}" with average evidence quality of {avg_evidence_quality:.1f}%.
            
            Key Findings:
            - {competitor_count} competitor sources and {third_party_count} third-party sources analyzed
            - Content quality shows {"strong" if avg_evidence_quality > 70 else "moderate" if avg_evidence_quality > 50 else "improvement needed"} evidence quality
            - {len(high_priority_recs)} high-priority recommendations identified
            
            Strategic Recommendations:
            Focus on the {len(high_priority_recs)} high-priority items to improve content performance and competitive positioning.
            
            Next Steps:
            Implement technical SEO optimizations and content improvements based on DNA analysis insights.
            
            Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """


def aggregate_pipeline_results(run_id: str, query: str, 
                             dna_results: List[Dict[str, Any]]) -> AggregationResult:
    """
    Main function to aggregate pipeline results.
    
    Args:
        run_id: Pipeline run identifier
        query: Original search query
        dna_results: List of DNA analysis results from Stage 2
        
    Returns:
        AggregationResult with comprehensive insights
    """
    aggregator = FinalAggregator()
    return aggregator.aggregate_run_results(run_id, query, dna_results)
