#!/usr/bin/env python3
"""
Test Perplexity workflow with existing test infrastructure
"""

import requests
import json
import time
import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment
load_dotenv()

# Initialize Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

BASE_URL = "http://localhost:8000"

def test_perplexity_workflow():
    """Test complete workflow for perplexity source"""
    print("🔄 Testing Perplexity Workflow")
    print("=" * 60)
    
    # Find a product with perplexity data
    try:
        result = supabase.table("product_analysis_perplexity")\
            .select("product_id, raw_serp_results")\
            .limit(5)\
            .execute()
        
        if not result.data:
            print("❌ No perplexity data found")
            return
        
        # Find first product with data
        test_product = None
        for item in result.data:
            raw_serp = item.get('raw_serp_results', {})
            if isinstance(raw_serp, dict) and raw_serp.get('source_links'):
                test_product = {
                    'product_id': item['product_id'],
                    'query': raw_serp.get('query', 'Unknown'),
                    'links_count': len(raw_serp.get('source_links', []))
                }
                break
        
        if not test_product:
            print("❌ No product with source_links found in perplexity data")
            return
        
        product_id = test_product['product_id']

        # IMPORTANT: product_id can have many-to-one input rows.
        # Count combined + de-duped source_links across ALL rows (matches orchestrator behavior).
        all_rows = supabase.table("product_analysis_perplexity")\
            .select("id, raw_serp_results")\
            .eq("product_id", product_id)\
            .execute()

        merged_links = []
        if all_rows.data:
            for row in all_rows.data:
                raw_serp = row.get('raw_serp_results', {})
                if isinstance(raw_serp, str):
                    try:
                        raw_serp = json.loads(raw_serp)
                    except Exception:
                        raw_serp = {}

                if isinstance(raw_serp, dict):
                    links = raw_serp.get('source_links') or []
                    if isinstance(links, list) and links:
                        merged_links.extend(links)

        unique_links = []
        seen = set()
        for link in merged_links:
            if isinstance(link, dict):
                key = link.get('raw_url') or link.get('url') or json.dumps(link, sort_keys=True)
            else:
                key = str(link)

            if key in seen:
                continue
            seen.add(key)
            unique_links.append(link)

        print(f"📦 Testing Product: {product_id}")
        print(f"📝 Query: {test_product['query']}")
        print(f"🔗 Source Links (first row): {test_product['links_count']}")
        print(f"🔗 Source Links (all rows, unique): {len(unique_links)}")
        print("-" * 40)
        
        # Test 1: Health check
        print("🏥 Step 1: Health Check")
        try:
            health_response = requests.get(f"{BASE_URL}/health")
            if health_response.ok:
                print(f"✅ Server healthy: {health_response.json()['status']}")
            else:
                print(f"❌ Server unhealthy: {health_response.status_code}")
                return
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return
        
        # Test 2: Process perplexity product
        print(f"\n🚀 Step 2: Processing Perplexity Product")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{BASE_URL}/process",
                json={
                    "product_id": product_id,
                    "source": "perplexity"
                },
                headers={"Content-Type": "application/json"}
            )
            
            processing_time = time.time() - start_time
            print(f"📡 Response Status: {response.status_code}")
            print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
            
            if response.ok:
                result = response.json()
                print(f"✅ Status: {result['status']}")
                print(f"📝 Message: {result.get('message', 'No message')}")
                
                if result['status'] == 'completed':
                    print(f"🎉 Full Pipeline Completed!")
                    print(f"📁 Run ID: {result.get('run_id')}")
                    print(f"🆔 Analysis ID: {result.get('analysis_id')}")
                    
                    # Test 3: Get detailed results
                    print(f"\n🔍 Step 3: Retrieving Results")
                    status_response = requests.post(
                        f"{BASE_URL}/status",
                        json={"product_id": product_id},
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if status_response.ok:
                        status = status_response.json()
                        blueprint = status.get('dna_blueprint', {})
                        
                        if isinstance(blueprint, dict) and blueprint.get('master_blueprint'):
                            master = blueprint['master_blueprint']
                            print(f"📊 Results Summary:")
                            print(f"   🔍 Query: {blueprint.get('query', 'N/A')}")
                            print(f"   🎯 Blueprint: ✅ Generated")
                            print(f"   📈 Top Performers: {len(master.get('top_performers', []))}")
                            print(f"   📋 Content Gaps: {len(master.get('content_gaps', []))}")
                            print(f"   💡 Recommendations: {len(master.get('recommendations', []))}")
                        else:
                            print(f"⚠️  No master blueprint found")
                            
                        print(f"📅 Created: {status.get('created_at')}")
                        print(f"🔄 Updated: {status.get('updated_at')}")
                        
                elif result['status'] == 'skipped':
                    print(f"⏭️  Processing Skipped: {result.get('message', 'Unknown reason')}")
                    
                elif result['status'] == 'already_exists':
                    print(f"ℹ️  Already Processed: {result.get('message', 'Unknown')}")
                    
                else:
                    print(f"❌ Processing Failed: {result.get('error', 'Unknown error')}")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Processing failed: {e}")
        
        print("\n" + "=" * 60)
        print("🎯 Perplexity Workflow Test Complete!")
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")

def test_batch_processing():
    """Test batch processing for perplexity"""
    print("\n🔄 Testing Perplexity Batch Processing")
    print("-" * 40)
    
    try:
        response = requests.post(
            f"{BASE_URL}/process-batch",
            json={
                "source": "perplexity",
                "limit": 3
            },
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📡 Batch Status: {response.status_code}")
        
        if response.ok:
            results = response.json()
            print(f"✅ Batch processed {len(results)} items")
            
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. {result['product_id']}: {result['status']}")
        else:
            print(f"❌ Batch failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch test failed: {e}")

def main():
    """Main test runner"""
    print("🧪 Perplexity Workflow Testing")
    print("Using existing test infrastructure")
    print("=" * 60)
    
    # Test single product processing
    test_perplexity_workflow()
    
    # Test batch processing
    test_batch_processing()
    
    print("\n🎉 All Perplexity Tests Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
