#!/usr/bin/env python3
"""
Find and process a product with actual data
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

def find_product_with_data(source="google"):
    """Find a product that has source_links"""
    try:
        # Get products with non-empty source_links
        result = supabase.table(f"product_analysis_{source}")\
            .select("product_id, source_links, query")\
            .execute()
        
        products = result.data
        print(f"📊 Found {len(products)} total products in {source}")
        
        # Filter for products with data
        products_with_data = []
        for product in products:
            source_links = product.get('source_links', [])
            if source_links and len(source_links) > 0:
                products_with_data.append(product)
        
        print(f"📊 Found {len(products_with_data)} products with data:")
        
        for i, product in enumerate(products_with_data[:3]):  # Show first 3
            print(f"   {i+1}. {product['product_id']}")
            print(f"      Query: {product.get('query', 'N/A')}")
            print(f"      Links: {len(product.get('source_links', []))}")
            print()
        
        if products_with_data:
            return products_with_data[0]['product_id']
        return None
        
    except Exception as e:
        print(f"❌ Error finding products: {e}")
        return None

def process_product_with_data():
    """Find and process a product with actual data"""
    BASE_URL = "http://localhost:8000"
    
    print("🔍 Finding product with data...")
    product_id = find_product_with_data("google")
    
    if not product_id:
        print("❌ No products with data found")
        return
    
    print(f"🚀 Processing product with data: {product_id}")
    print("=" * 60)
    
    # Process the product
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
            print(f"✅ Status: {result['status']}")
            
            if result['status'] == 'completed':
                print(f"🎉 Full pipeline completed!")
                print(f"📁 Run ID: {result.get('run_id')}")
                print(f"🆔 Analysis ID: {result.get('analysis_id')}")
                print(f"📂 Output: {result.get('final_output_path')}")
                
                # Get detailed results
                status_response = requests.post(
                    f"{BASE_URL}/status",
                    json={"product_id": product_id},
                    headers={"Content-Type": "application/json"}
                )
                
                if status_response.ok:
                    status = status_response.json()
                    blueprint = status.get('dna_blueprint', {})
                    
                    if blueprint and isinstance(blueprint, dict):
                        print(f"\n📊 Results Summary:")
                        if blueprint.get('query'):
                            print(f"   🔍 Query: {blueprint['query']}")
                        
                        if blueprint.get('master_blueprint'):
                            master = blueprint['master_blueprint']
                            print(f"   🎯 Master Blueprint Found!")
                            print(f"      - Top Performers: {len(master.get('top_performers', []))}")
                            print(f"      - Content Gaps: {len(master.get('content_gaps', []))}")
                            print(f"      - Recommendations: {len(master.get('recommendations', []))}")
                            
                            # Show sample content
                            if master.get('top_performers'):
                                performer = master['top_performers'][0]
                                print(f"      - Sample URL: {performer.get('url', 'N/A')[:80]}...")
                            
                            if master.get('recommendations'):
                                rec = master['recommendations'][0]
                                print(f"      - Sample Rec: {rec.get('title', 'N/A')}")
                        else:
                            print(f"   ⚠️  No master blueprint (check API key)")
                    else:
                        print(f"   📄 Blueprint: {type(blueprint)}")
                        
            elif result['status'] == 'already_exists':
                print(f"ℹ️  Already processed - checking results...")
                # Get existing results
                status_response = requests.post(
                    f"{BASE_URL}/status",
                    json={"product_id": product_id},
                    headers={"Content-Type": "application/json"}
                )
                
                if status_response.ok:
                    status = status_response.json()
                    print(f"📊 Existing Status: {status['status']}")
                    print(f"📅 Created: {status.get('created_at')}")
                    
            else:
                print(f"❌ Failed: {result.get('error')}")
                
        else:
            print(f"❌ HTTP Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    process_product_with_data()
