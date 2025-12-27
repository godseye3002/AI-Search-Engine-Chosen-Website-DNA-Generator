#!/usr/bin/env python3
"""
Process a fresh product through the entire pipeline
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def process_fresh_product():
    """Process a product that hasn't been processed yet"""
    
    # Try a different product ID from the batch results
    fresh_product_id = "9730f5ba-8b22-495a-a577-46432a9f060f"
    source = "google"
    
    print(f"🚀 Processing Fresh Product: {fresh_product_id}")
    print("=" * 60)
    
    # Step 1: Process the product
    print(f"🔄 Step 1: Starting pipeline processing...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/process",
            json={
                "product_id": fresh_product_id,
                "source": source
            },
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.ok:
            result = response.json()
            print(f"✅ Process Status: {result['status']}")
            
            if result['status'] == 'completed':
                print(f"🎉 SUCCESS! Full pipeline completed!")
                print(f"📁 Run ID: {result.get('run_id')}")
                print(f"🆔 Analysis ID: {result.get('analysis_id')}")
                print(f"📂 Output Path: {result.get('final_output_path')}")
                print(f"⏱️  Processing Time: {time.time() - start_time:.2f} seconds")
                
                # Step 2: Get the full results
                print(f"\n🔍 Step 2: Retrieving complete results...")
                
                status_response = requests.post(
                    f"{BASE_URL}/status",
                    json={"product_id": fresh_product_id},
                    headers={"Content-Type": "application/json"}
                )
                
                if status_response.ok:
                    status_result = status_response.json()
                    
                    # Check DNA blueprint
                    if status_result.get('dna_blueprint'):
                        blueprint = status_result['dna_blueprint']
                        print(f"📊 DNA Blueprint Retrieved!")
                        
                        if isinstance(blueprint, dict):
                            if blueprint.get('query'):
                                print(f"   🔍 Query: {blueprint['query']}")
                            
                            if blueprint.get('master_blueprint'):
                                master = blueprint['master_blueprint']
                                print(f"   🎯 Master Blueprint:")
                                print(f"      - Top Performers: {len(master.get('top_performers', []))}")
                                print(f"      - Content Gaps: {len(master.get('content_gaps', []))}")
                                print(f"      - Recommendations: {len(master.get('recommendations', []))}")
                                
                                # Show some details
                                if master.get('top_performers'):
                                    print(f"      - Sample Top Performer: {master['top_performers'][0].get('url', 'N/A')[:80]}...")
                                if master.get('recommendations'):
                                    print(f"      - Sample Recommendation: {master['recommendations'][0].get('title', 'N/A')}")
                            else:
                                print(f"   ⚠️  No master blueprint found (may be null due to API key issues)")
                        else:
                            print(f"   📄 Blueprint Type: {type(blueprint)}")
                    else:
                        print(f"   ❌ No DNA blueprint found")
                        
                    print(f"📅 Created: {status_result.get('created_at')}")
                    print(f"🔄 Updated: {status_result.get('updated_at')}")
                else:
                    print(f"❌ Failed to get status: {status_response.status_code}")
                    
            elif result['status'] == 'skipped':
                print(f"⏭️  Product was skipped (likely empty source_links)")
                print(f"📝 Reason: {result.get('message', 'Unknown')}")
                
            else:
                print(f"❌ Processing failed: {result.get('error')}")
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"📄 Error Details: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 60)

def main():
    """Main runner"""
    print("🧪 Testing Full Pipeline with Fresh Product")
    print("This will run Stage 1 → Stage 2 → Stage 3 → Database Save")
    print("=" * 60)
    
    # Health check
    try:
        health = requests.get(f"{BASE_URL}/health")
        if not health.ok:
            print("❌ Server not ready")
            return
        print("✅ Server ready")
    except:
        print("❌ Cannot connect to server")
        return
    
    # Process fresh product
    process_fresh_product()

if __name__ == "__main__":
    main()
