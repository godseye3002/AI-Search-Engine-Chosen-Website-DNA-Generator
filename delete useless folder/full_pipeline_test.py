#!/usr/bin/env python3
"""
Process product with raw_serp_results data
"""

import requests
import json
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

def find_product_with_serp_data():
    """Find product with raw_serp_results containing source_links"""
    try:
        result = supabase.table("product_analysis_google")\
            .select("product_id, raw_serp_results")\
            .execute()
        
        products_with_data = []
        
        for product in result.data:
            raw_serp = product.get('raw_serp_results', {})
            
            # Check if raw_serp_results has source_links
            if isinstance(raw_serp, dict):
                source_links = raw_serp.get('source_links', [])
                if source_links and len(source_links) > 0:
                    products_with_data.append({
                        'product_id': product['product_id'],
                        'query': raw_serp.get('query', 'Unknown'),
                        'links_count': len(source_links),
                        'raw_serp': raw_serp
                    })
        
        print(f"🎯 Found {len(products_with_data)} products with SERP data:")
        
        for i, product in enumerate(products_with_data[:3]):
            print(f"   {i+1}. {product['product_id']}")
            print(f"      Query: {product['query']}")
            print(f"      Links: {product['links_count']}")
            print()
        
        return products_with_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def run_full_pipeline_test():
    """Run the complete pipeline test"""
    BASE_URL = "http://localhost:8000"
    
    print("🚀 GodsEye DNA Pipeline - Full Workflow Test")
    print("=" * 60)
    
    # Find products with data
    products = find_product_with_serp_data()
    
    if not products:
        print("❌ No products with SERP data found")
        return
    
    # Process the first product
    product = products[0]
    product_id = product['product_id']
    
    print(f"🔄 Processing: {product_id}")
    print(f"📝 Query: {product['query']}")
    print(f"🔗 Source Links: {product['links_count']}")
    print("-" * 40)
    
    # Step 1: Process via API
    print("🚀 Step 1: Starting pipeline...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/process",
            json={
                "product_id": product_id,
                "source": "google"
            },
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.ok:
            result = response.json()
            processing_time = time.time() - start_time
            
            print(f"✅ Processing Status: {result['status']}")
            print(f"⏱️  Time: {processing_time:.2f} seconds")
            
            if result['status'] == 'completed':
                print(f"🎉 FULL PIPELINE COMPLETED!")
                print(f"📁 Run ID: {result.get('run_id')}")
                print(f"🆔 Analysis ID: {result.get('analysis_id')}")
                print(f"📂 Output: {result.get('final_output_path')}")
                
                # Step 2: Get complete results
                print(f"\n🔍 Step 2: Retrieving DNA blueprint...")
                
                status_response = requests.post(
                    f"{BASE_URL}/status",
                    json={"product_id": product_id},
                    headers={"Content-Type": "application/json"}
                )
                
                if status_response.ok:
                    status = status_response.json()
                    blueprint = status.get('dna_blueprint', {})
                    
                    print(f"📊 DNA Blueprint Results:")
                    
                    if isinstance(blueprint, dict):
                        if blueprint.get('query'):
                            print(f"   🔍 Original Query: {blueprint['query']}")
                        
                        if blueprint.get('master_blueprint'):
                            master = blueprint['master_blueprint']
                            print(f"   ✅ Master Blueprint Generated!")
                            print(f"   📈 Top Performers: {len(master.get('top_performers', []))}")
                            print(f"   📋 Content Gaps: {len(master.get('content_gaps', []))}")
                            print(f"   💡 Recommendations: {len(master.get('recommendations', []))}")
                            
                            # Show sample results
                            if master.get('top_performers'):
                                performer = master['top_performers'][0]
                                print(f"\n   🏆 Sample Top Performer:")
                                print(f"      URL: {performer.get('url', 'N/A')}")
                                print(f"      Score: {performer.get('overall_score', 'N/A')}")
                            
                            if master.get('recommendations'):
                                rec = master['recommendations'][0]
                                print(f"\n   💡 Sample Recommendation:")
                                print(f"      Title: {rec.get('title', 'N/A')}")
                                print(f"      Priority: {rec.get('priority', 'N/A')}")
                                
                            print(f"\n   🎯 Pipeline Success: All 3 stages completed!")
                        else:
                            print(f"   ⚠️  Master blueprint is null or missing")
                            print(f"   🔧 This usually means Gemini API key issue")
                            
                    else:
                        print(f"   📄 Blueprint Type: {type(blueprint)}")
                        
                    print(f"\n📅 Analysis Created: {status.get('created_at')}")
                    print(f"🔄 Last Updated: {status.get('updated_at')}")
                    
                else:
                    print(f"❌ Failed to get status: {status_response.status_code}")
                    
            elif result['status'] == 'already_exists':
                print(f"ℹ️  Product already processed")
                # Get existing results
                status_response = requests.post(
                    f"{BASE_URL}/status",
                    json={"product_id": product_id},
                    headers={"Content-Type": "application/json"}
                )
                
                if status_response.ok:
                    status = status_response.json()
                    print(f"📊 Status: {status['status']}")
                    
            elif result['status'] == 'skipped':
                print(f"⏭️  Product skipped (empty source_links)")
                
            else:
                print(f"❌ Processing failed: {result.get('error')}")
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"📄 Details: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Full workflow test completed!")

if __name__ == "__main__":
    import time
    run_full_pipeline_test()
