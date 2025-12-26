"""
Simplified Final Aggregation Core - Matches master_blueprint_dna_generator output format
Generates only the master blueprint JSON structure with verbose logging.
"""

import json
import os
import logging
from typing import Any, Dict, List
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv
from utils.env_utils import is_production_mode, get_log_level

# Configure logging based on environment
log_level = logging.INFO if not is_production_mode() else logging.ERROR
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - [STAGE3] - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv() 
API_KEY = os.getenv('GEMINI_API_KEY', '')
genai.configure(api_key=API_KEY)


class SimpleBlueprintGenerator:
    """Simplified blueprint generator matching master_blueprint_dna_generator output"""
    
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
    
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        if not is_production_mode():
            logger.info(f"Initializing SimpleBlueprintGenerator with model: {model_name}")
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        if not is_production_mode():
            logger.info("Gemini model configured successfully")
    
    def generate_master_blueprint(self, dna_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Generate master blueprint from DNA analysis results
        
        Args:
            dna_results: List of DNA analysis results from Stage 2
            query: Original search query
            
        Returns:
            Master blueprint JSON structure
        """
        if not is_production_mode():
            logger.info(f"Starting master blueprint generation for {len(dna_results)} DNA results")
            logger.info(f"Query context: {query}")
        
        try:
            # Prepare input for Gemini
            input_text = f"Query: {query}\n\nDNA Analysis Results:\n"
            for i, result in enumerate(dna_results, 1):
                input_text += f"\n--- Source {i} ---\n{json.dumps(result, indent=2)}\n"
            
            if not is_production_mode():
                logger.info(f"Prepared input text of length: {len(input_text)} characters")
            
            # Generate blueprint
            if not is_production_mode():
                logger.info("Calling Gemini API to generate master blueprint...")
            response = self.model.generate_content(self.MASTER_BLUEPRINT_PROMPT + "\n\n" + input_text)
            response_text = response.text
            
            if not is_production_mode():
                logger.info(f"Received response of length: {len(response_text)} characters")
            
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
            if not is_production_mode():
                logger.info("Parsing JSON response...")
            master_blueprint = json.loads(response_text)
            
            # Add metadata
            master_blueprint['generation_timestamp'] = datetime.now().isoformat()
            master_blueprint['sources_processed'] = len(dna_results)
            
            if not is_production_mode():
                logger.info(f"Successfully generated master blueprint with {len(master_blueprint)} top-level keys")
            return master_blueprint
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            if not is_production_mode():
                logger.debug(f"Response text: {response_text[:500]}...")
            return None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating master blueprint: {error_msg}")
            
            # Check for token limit error and implement fallback
            if "generation exceeded max tokens limit" in error_msg or "max_output_tokens" in error_msg:
                if not is_production_mode():
                    logger.warning("Token limit exceeded, trying with reduced input size...")
                return self._generate_with_reduced_input(dna_results, query)
            
            return None
    
    def _generate_with_reduced_input(self, dna_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Fallback method with reduced input size to handle token limits"""
        if not is_production_mode():
            logger.info("Attempting generation with reduced input size...")
        
        try:
            # Take only the first 3 results to reduce input size
            reduced_results = dna_results[:3]
            if not is_production_mode():
                logger.info(f"Reduced input from {len(dna_results)} to {len(reduced_results)} results")
            
            # Prepare simplified input
            input_text = f"Query: {query}\n\nDNA Analysis Results (Top 3):\n"
            for i, result in enumerate(reduced_results, 1):
                # Only include essential fields to further reduce size
                simplified_result = {
                    'dna_analysis': result.get('dna_analysis', {}),
                    'classification': result.get('classification', ''),
                    'url': result.get('url', '')
                }
                input_text += f"\n--- Source {i} ---\n{json.dumps(simplified_result, indent=2)}\n"
            
            if not is_production_mode():
                logger.info(f"Prepared reduced input text of length: {len(input_text)} characters")
            # gemini-2.5-pro
            # gemini-3-pro-preview
            # Generate with even smaller token limit
            reduced_model = genai.GenerativeModel(
                model_name="gemini-2.5-pro",
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 4048,  # Further reduced
                }
            )
            
            logger.info("Calling Gemini API with reduced settings...")
            response = reduced_model.generate_content(self.MASTER_BLUEPRINT_PROMPT + "\n\n" + input_text)
            response_text = response.text
            
            logger.info(f"Received reduced response of length: {len(response_text)} characters")
            
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
            logger.info("Parsing JSON response from reduced input...")
            master_blueprint = json.loads(response_text)
            
            # Add metadata indicating fallback was used
            master_blueprint['generation_timestamp'] = datetime.now().isoformat()
            master_blueprint['sources_processed'] = len(reduced_results)
            master_blueprint['fallback_used'] = True
            master_blueprint['original_sources_count'] = len(dna_results)
            
            logger.info(f"Successfully generated master blueprint with fallback using {len(reduced_results)} sources")
            return master_blueprint
            
        except Exception as e:
            logger.error(f"Fallback generation also failed: {str(e)}")
            return None


def aggregate_pipeline_results(run_id: str, query: str, 
                             dna_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to aggregate pipeline results - simplified output format.
    
    Args:
        run_id: Pipeline run identifier
        query: Original search query
        dna_results: List of DNA analysis results from Stage 2
        
    Returns:
        Master blueprint JSON structure (matches master_blueprint_dna_generator format)
    """
    logger.info(f"Starting pipeline aggregation for run: {run_id}")
    logger.info(f"Processing {len(dna_results)} DNA results")
    
    try:
        generator = SimpleBlueprintGenerator()
        master_blueprint = generator.generate_master_blueprint(dna_results, query)
        
        if master_blueprint:
            logger.info(f"Successfully generated master blueprint for run {run_id}")
            return master_blueprint
        else:
            logger.error(f"Failed to generate master blueprint for run {run_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error in aggregate_pipeline_results: {str(e)}")
        return None
