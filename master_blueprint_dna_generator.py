"""
Master Blueprint Generator
Processes winning content profiles from Google AI Overview and Perplexity
using Gemini LLM to generate unified master blueprints.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import google.generativeai as genai
from dataclasses import dataclass, asdict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Data class for processing results"""
    source_type: str
    success: bool
    input_files_count: int
    processed_files: List[str]
    output_file: Optional[str]
    error_message: Optional[str]
    processing_time: float
    timestamp: str


class MasterBlueprintGenerator:
    """Main class for generating master blueprints from content profiles"""
    
    MASTER_ARCHITECT_PROMPT = """
**Role:** Master Information Architect & AEO Synthesis Engine
**Task:** Aggregate multiple "Winning Content Profiles" (JSON) into a single, high-efficiency "Master Blueprint."
**Goal:** Create a unified data structure that holds the *maximum* amount of unique, high-value information with the *minimum* amount of redundancy, optimized for Generative Engine Optimization (GEO).

---

### **Context & Objective**
You are an advanced data synthesizer. Your job is to ingest N number of JSON files. Each file represents a high-ranking webpage (a "source model"). Your goal is to combine them into one **"Master Blueprint JSON"**.

This Master Blueprint must not be a simple concatenation. It must be a **smart reconstruction** that follows these core engineering principles:
1.  **Structural Consensus:** Identify the most common HTML layout patterns (e.g., H1 followed by P, then H2) and establish them as the `core_layout_schema`.
2.  **Entity Deduplication:** If "Entity_Alpha" appears in 10 files, store it *once* as a primary entity.
3.  **Information Gain Maximization (The Golden Rule):**
    * **Consensus Data (Low Entropy):** If 8/10 sources describe an entity generically (e.g., "Standard Industry Solution"), store this as the `primary_definition`.
    * **Unique Data (High Entropy):** If only 1/10 sources provides a specific statistic (e.g., "Value_X" or "Percentage_Y"), you **MUST** preserve this as a `high_value_attribute`. Do not discard rare data if it contains numbers, percentages, or proper nouns.
4.  **Noise Filtration:** Discard generic fluff (e.g., "Best in class," "Amazing performance") that carries no factual weight.

---

### **The Logic: How to Process the Input**

**Phase 1: Structural Analysis (The Skeleton)**
Scan the `dominant_html_patterns` of all inputs. Determine the "Winning Sequence."
* *Logic:* If the majority of successful pages use a "Comparison Module" after the "Introduction," your Master Blueprint must enforce a `required_module: comparison_module` at that position.

**Phase 2: Entity Fusion (The Data)**
Create a centralized database of all entities (products, concepts, people) found across the inputs.
* **Merge Strategy:**
    * *Input A:* "Entity_Alpha is a Type_1 tool."
    * *Input B:* "Entity_Alpha solves Problem_X."
    * *Input C:* "Entity_Alpha has Feature_Z."
    * *Result:* Store `Entity: Entity_Alpha` with `attributes: ["Type_1 tool", "Solves Problem_X", "Feature_Z"]`.

**Phase 3: Validation Layers (The Quality Control)**
Identify the mandatory elements required for the content to be considered "Complete" by a search engine.
* *Logic:* If high-ranking pages always cite "Regulatory Standard A" when discussing "Topic B," create a rule: `dependency: Topic B requires Regulatory_Standard_A`.

---

### **Target Output Format (JSON Schema)**

You must output the final result in the following JSON structure. **Do not use biological terms.** Use the keys exactly as defined below:

```json
{
  "master_blueprint_id": "aggregated_v1",
  "blueprint_generation_logic": "Consensus_Plus_Uniqueness",
  
  "layer_1_structural_framework": [
    {
      "sequence_order": 1,
      "component_type": "h1_header",
      "consensus_intent": "Define the Topic clearly",
      "optimized_pattern_template": "The [Adjective] Guide to [Topic] for [Audience]"
    },
    {
      "sequence_order": 2,
      "component_type": "summary_block",
      "content_requirement": "Must include key entities: [List_Top_Entities_Found]"
    }
    // Continue for the ideal page structure...
  ],

  "layer_2_entity_knowledge_base": {
    "Entity_Name_Here": {
      "classification": "e.g., Primary Category / Tool Type",
      "primary_definition": "The most common description found across sources.",
      "high_value_attributes": [
        {
          "data_point": "Specific statistic or unique fact (e.g., 'Has 500+ integrations')",
          "source_weight": "High (derived from unique signal)"
        },
        {
          "data_point": "Specific feature (e.g., 'Automated Workflow Support')",
          "source_weight": "Medium"
        }
      ],
      "semantic_keywords": ["List", "Related", "Keywords"]
    }
    // Repeat for all key entities found...
  },

  "layer_3_compliance_protocols": {
    "mandatory_inclusions": [
      "List of items that MUST be present to avoid low-quality scoring"
    ],
    "negative_constraints": [
      "List of patterns to avoid (e.g., 'Generic Intros')"
    ]
  }
}
```

---

**CRITICAL INSTRUCTIONS:**
1. Analyze ALL provided JSON files
2. Generate output in STRICT JSON format only
3. No markdown, no explanations outside the JSON
4. Follow the exact schema provided above
5. Preserve all unique high-value data points
"""

    def __init__(self, api_key: str, model_name: str = "gemini-3-pro-preview"):
        """
        Initialize the generator with Gemini API credentials
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self._configure_gemini()
        
    def _configure_gemini(self):
        """Configure Gemini API"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            logger.info(f"Gemini API configured successfully with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}")
            raise
    
    def load_json_files(self, directory: str, file_pattern: str = "*.json") -> Tuple[List[Dict], List[str]]:
        """
        Load JSON files from a directory
        
        Args:
            directory: Directory path containing JSON files
            file_pattern: Pattern to match JSON files
            
        Returns:
            Tuple of (list of JSON data, list of filenames)
        """
        json_data = []
        filenames = []
        
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return json_data, filenames
        
        json_files = list(dir_path.glob(file_pattern))
        logger.info(f"Found {len(json_files)} JSON files in {directory}")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    json_data.append(data)
                    filenames.append(file_path.name)
                    logger.info(f"Successfully loaded: {file_path.name}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {file_path.name}: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {str(e)}")
        
        return json_data, filenames
    
    def generate_master_blueprint(self, json_data_list: List[Dict], source_type: str) -> Optional[Dict]:
        """
        Generate master blueprint using Gemini LLM
        
        Args:
            json_data_list: List of JSON data from content profiles
            source_type: Source type (google_ai_overview or perplexity)
            
        Returns:
            Master blueprint as dictionary or None if failed
        """
        if not json_data_list:
            logger.warning(f"No JSON data provided for {source_type}")
            return None
        
        try:
            # Prepare the input data
            input_data = {
                "source_type": source_type,
                "total_files": len(json_data_list),
                "content_profiles": json_data_list
            }
            
            # Create the full prompt
            full_prompt = f"{self.MASTER_ARCHITECT_PROMPT}\n\n**INPUT DATA:**\n```json\n{json.dumps(input_data, indent=2)}\n```"
            
            logger.info(f"Generating master blueprint for {source_type} with {len(json_data_list)} files...")
            
            # Generate content using Gemini
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
            logger.info(f"Successfully generated master blueprint for {source_type}")
            
            return master_blueprint
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for {source_type}: {str(e)}")
            logger.debug(f"Response text: {response_text[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Error generating master blueprint for {source_type}: {str(e)}")
            return None
    
    def save_output(self, data: Dict, output_path: str) -> bool:
        """
        Save output to JSON file
        
        Args:
            data: Data to save
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved output to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving output to {output_path}: {str(e)}")
            return False
    
    def process_source(
        self, 
        source_type: str, 
        input_directory: str, 
        output_directory: str
    ) -> ProcessingResult:
        """
        Process a single source (Google AI Overview or Perplexity)
        
        Args:
            source_type: Type of source (google_ai_overview or perplexity)
            input_directory: Directory containing input JSON files
            output_directory: Directory to save output
            
        Returns:
            ProcessingResult object with processing details
        """
        start_time = datetime.now()
        
        try:
            # Load JSON files
            json_data_list, filenames = self.load_json_files(input_directory)
            
            if not json_data_list:
                return ProcessingResult(
                    source_type=source_type,
                    success=False,
                    input_files_count=0,
                    processed_files=[],
                    output_file=None,
                    error_message="No valid JSON files found",
                    processing_time=0.0,
                    timestamp=datetime.now().isoformat()
                )
            
            # Generate master blueprint
            master_blueprint = self.generate_master_blueprint(json_data_list, source_type)
            
            if master_blueprint is None:
                return ProcessingResult(
                    source_type=source_type,
                    success=False,
                    input_files_count=len(json_data_list),
                    processed_files=filenames,
                    output_file=None,
                    error_message="Failed to generate master blueprint",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    timestamp=datetime.now().isoformat()
                )
            
            # Save output
            output_filename = f"{source_type}_master_blueprint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = os.path.join(output_directory, output_filename)
            
            save_success = self.save_output(master_blueprint, output_path)
            
            if not save_success:
                return ProcessingResult(
                    source_type=source_type,
                    success=False,
                    input_files_count=len(json_data_list),
                    processed_files=filenames,
                    output_file=None,
                    error_message="Failed to save output file",
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    timestamp=datetime.now().isoformat()
                )
            
            return ProcessingResult(
                source_type=source_type,
                success=True,
                input_files_count=len(json_data_list),
                processed_files=filenames,
                output_file=output_path,
                error_message=None,
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Unexpected error processing {source_type}: {str(e)}")
            return ProcessingResult(
                source_type=source_type,
                success=False,
                input_files_count=0,
                processed_files=[],
                output_file=None,
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat()
            )


def main():
    """Main execution function"""
    
    # Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
    
    # Directory structure
    BASE_INPUT_DIR = "input"
    BASE_OUTPUT_DIR = "output"
    
    GOOGLE_AI_INPUT_DIR = os.path.join(BASE_INPUT_DIR, "google_ai_overview")
    PERPLEXITY_INPUT_DIR = os.path.join(BASE_INPUT_DIR, "perplexity")
    
    GOOGLE_AI_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "google_ai_overview")
    PERPLEXITY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "perplexity")
    
    # Create directories if they don't exist
    for directory in [GOOGLE_AI_INPUT_DIR, PERPLEXITY_INPUT_DIR, 
                      GOOGLE_AI_OUTPUT_DIR, PERPLEXITY_OUTPUT_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MASTER BLUEPRINT GENERATOR - STARTING")
    logger.info("=" * 80)
    
    # Initialize generator
    try:
        generator = MasterBlueprintGenerator(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize generator: {str(e)}")
        return
    
    # Process both sources
    results = []
    
    # Process Google AI Overview
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING: Google AI Overview")
    logger.info("=" * 80)
    google_result = generator.process_source(
        source_type="google_ai_overview",
        input_directory=GOOGLE_AI_INPUT_DIR,
        output_directory=GOOGLE_AI_OUTPUT_DIR
    )
    results.append(google_result)
    
    # Process Perplexity
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING: Perplexity")
    logger.info("=" * 80)
    perplexity_result = generator.process_source(
        source_type="perplexity",
        input_directory=PERPLEXITY_INPUT_DIR,
        output_directory=PERPLEXITY_OUTPUT_DIR
    )
    results.append(perplexity_result)
    
    # Generate and save processing report
    report = {
        "execution_timestamp": datetime.now().isoformat(),
        "total_sources_processed": len(results),
        "results": [asdict(result) for result in results]
    }
    
    report_path = os.path.join(BASE_OUTPUT_DIR, f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    generator.save_output(report, report_path)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    for result in results:
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        logger.info(f"\n{result.source_type.upper()}: {status}")
        logger.info(f"  Input Files: {result.input_files_count}")
        logger.info(f"  Processed: {', '.join(result.processed_files) if result.processed_files else 'None'}")
        logger.info(f"  Output: {result.output_file if result.output_file else 'N/A'}")
        logger.info(f"  Time: {result.processing_time:.2f}s")
        if result.error_message:
            logger.error(f"  Error: {result.error_message}")
    
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Full report saved to: {report_path}")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()