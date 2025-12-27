#!/usr/bin/env python3
"""
Full workflow test for the GodsEye DNA Pipeline API
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def get_available_products(source):
    """Get available products from the database"""
    try:
        # This would require a new endpoint to list products
        # For now, let's use the known product ID
        return ["02f92e70-7b53-45b6-bdef-7ef36d8fc578"]
    except Exception as e:
        print(f"Error getting products: {e}")
        return []

def run_full_workflow():
    """Run the complete workflow for both sources"""
    print("🚀 Starting GodsEye DNA Pipeline Full Workflow Test")
    print("=" * 60)
    
    # Test both sources
    sources = ["google", "perplexity"]
    product_id = "02f92e70-7b53-45b6-bdef-7ef36d8fc578"
    
    for source in sources:
        print(f"\n📊 Processing {source.upper()} source...")
        print("-" * 40)
        
        # Step 1: Process the product
        print(f"🔄 Step 1: Processing product {product_id} from {source}")
        try:
            response = requests.post(
                f"{BASE_URL}/process",
                json={
                    "product_id": product_id,
                    "source": source
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.ok:
                result = response.json()
                print(f"✅ Process Status: {result['status']}")
                
                if result['status'] == 'completed':
                    print(f"📁 Run ID: {result.get('run_id')}")
                    print(f"🆔 Analysis ID: {result.get('analysis_id')}")
                    print(f"📂 Output Path: {result.get('final_output_path')}")
                elif result['status'] == 'already_exists':
                    print(f"ℹ️  Product already processed")
                    print(f"🆔 Analysis ID: {result.get('analysis_id')}")
                else:
                    print(f"❌ Processing failed: {result.get('error')}")
                    continue
                    
            else:
                print(f"❌ HTTP Error: {response.status_code} - {response.text}")
                continue
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            continue
        
        # Step 2: Check status and get results
        print(f"\n🔍 Step 2: Checking status for {product_id}")
        try:
            status_response = requests.post(
                f"{BASE_URL}/status",
                json={"product_id": product_id},
                headers={"Content-Type": "application/json"}
            )
            
            if status_response.ok:
                status_result = status_response.json()
                print(f"✅ Status: {status_result['status']}")
                print(f"📊 Data Source: {status_result['data_source']}")
                
                # Check if we have DNA blueprint
                if status_result.get('dna_blueprint'):
                    blueprint = status_result['dna_blueprint']
                    if isinstance(blueprint, dict) and blueprint.get('master_blueprint'):
                        print(f"🎯 Master Blueprint Found: ✅")
                        print(f"   - Query: {blueprint.get('query', 'N/A')}")
                        print(f"   - Top Performers: {len(blueprint.get('master_blueprint', {}).get('top_performers', []))}")
                        print(f"   - Content Gaps: {len(blueprint.get('master_blueprint', {}).get('content_gaps', []))}")
                        print(f"   - Recommendations: {len(blueprint.get('master_blueprint', {}).get('recommendations', []))}")
                    else:
                        print(f"⚠️  Blueprint exists but may be incomplete")
                else:
                    print(f"⚠️  No DNA blueprint found")
                    
                print(f"📅 Created: {status_result.get('created_at', 'N/A')}")
                print(f"🔄 Updated: {status_result.get('updated_at', 'N/A')}")
            else:
                print(f"❌ Status check failed: {status_response.status_code}")
                
        except Exception as e:
            print(f"❌ Status check failed: {e}")
        
        print("\n" + "=" * 60)

def test_batch_processing():
    """Test batch processing"""
    print("\n🔄 Testing Batch Processing...")
    print("-" * 40)
    
    try:
        response = requests.post(
            f"{BASE_URL}/process-batch",
            json={
                "source": "google",
                "limit": 5
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.ok:
            results = response.json()
            print(f"✅ Batch processed {len(results)} items")
            for i, result in enumerate(results[:3]):  # Show first 3
                print(f"   {i+1}. {result['product_id']}: {result['status']}")
        else:
            print(f"❌ Batch failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Batch test failed: {e}")

def get_pipeline_stats():
    """Get pipeline statistics"""
    print("\n📊 Pipeline Statistics")
    print("-" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.ok:
            stats = response.json()
            print(f"✅ Statistics retrieved:")
            for key, value in stats.get('data', {}).items():
                print(f"   {key}: {value}")
        else:
            print(f"❌ Stats failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Stats failed: {e}")

def main():
    """Main test runner"""
    start_time = time.time()
    
    # Health check first
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        if not health_response.ok:
            print("❌ Server not healthy - aborting test")
            return
        print("✅ Server is healthy")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the server is running: python serverless_api.py")
        return
    
    # Run full workflow
    run_full_workflow()
    
    # Test batch processing
    test_batch_processing()
    
    # Get statistics
    get_pipeline_stats()
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n🎉 Full workflow test completed in {duration:.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    main()
