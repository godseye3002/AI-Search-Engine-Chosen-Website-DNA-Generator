#!/usr/bin/env python3
"""
Test single pipeline run with proper website URLs
"""

import logging
from pipeline_orchestrator import PipelineOrchestrator

def main():
    logging.basicConfig(level=logging.INFO)
    orchestrator = PipelineOrchestrator()
    
    # Test with proper website URLs
    ai_response = orchestrator.load_ai_response('test_ai_response.json')
    print(f'Loaded {len(ai_response.get("source_links", []))} source links')
    print('Sample URLs:')
    for i, link in enumerate(ai_response.get('source_links', [])[:3]):
        print(f'  {i+1}. {link.get("url", "N/A")}')
    
    try:
        run_id = orchestrator.create_run(ai_response)
        print(f'Created run: {run_id}')
        
        orchestrator.run_pipeline('test_ai_response.json')
        print('Pipeline completed successfully!')
        
    except Exception as e:
        print(f'Pipeline failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
