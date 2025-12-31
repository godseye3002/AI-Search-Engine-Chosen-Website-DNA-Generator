#!/usr/bin/env python3
"""
Check database schema and find products with data
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

def check_schema():
    """Check what columns exist in the table"""
    try:
        # Get table info
        result = supabase.table("product_analysis_google").select("*").limit(1).execute()
        
        if result.data:
            columns = list(result.data[0].keys())
            print("📊 Columns in product_analysis_google:")
            for col in columns:
                print(f"   - {col}")
            
            # Check for similar column names
            link_columns = [col for col in columns if 'link' in col.lower() or 'source' in col.lower()]
            print(f"\n🔗 Link-related columns: {link_columns}")
            
        return result.data
        
    except Exception as e:
        print(f"❌ Error checking schema: {e}")
        return None

def find_products_with_data():
    """Find products with actual data"""
    try:
        # Get all products
        result = supabase.table("product_analysis_google").select("*").limit(10).execute()
        
        if not result.data:
            print("❌ No data found")
            return
        
        print(f"📊 Found {len(result.data)} products:")
        
        for i, product in enumerate(result.data[:5]):
            print(f"\n   {i+1}. Product ID: {product.get('product_id', 'N/A')}")
            print(f"      Query: {product.get('query', 'N/A')}")
            
            # Check all fields for data
            for key, value in product.items():
                if key != 'product_id' and value:
                    if isinstance(value, list) and len(value) > 0:
                        print(f"      {key}: {len(value)} items")
                    elif isinstance(value, str) and len(value) > 50:
                        print(f"      {key}: {value[:50]}...")
                    elif not isinstance(value, (str, list)):
                        print(f"      {key}: {type(value)} = {str(value)[:50]}")
        
        # Find products with lists/arrays
        products_with_lists = []
        for product in result.data:
            for key, value in product.items():
                if isinstance(value, list) and len(value) > 0:
                    products_with_lists.append(product)
                    break
        
        print(f"\n🎯 Found {len(products_with_lists)} products with list data:")
        for i, product in enumerate(products_with_lists[:3]):
            print(f"   {i+1}. {product.get('product_id', 'N/A')}")
            
        return products_with_lists
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def process_via_api(product_id):
    """Process a product via the API"""
    BASE_URL = "http://localhost:8000"
    
    print(f"\n🚀 Processing via API: {product_id}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/process",
            json={
                "product_id": product_id,
                "source": "google"
            },
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📡 Status: {response.status_code}")
        
        if response.ok:
            result = response.json()
            print(f"✅ Result: {result['status']}")
            
            if result['status'] == 'completed':
                print(f"🎉 Full pipeline completed!")
                print(f"📁 Run ID: {result.get('run_id')}")
                print(f"🆔 Analysis ID: {result.get('analysis_id')}")
                
                # Get detailed results
                status_resp = requests.post(
                    f"{BASE_URL}/status",
                    json={"product_id": product_id},
                    headers={"Content-Type": "application/json"}
                )
                
                if status_resp.ok:
                    status = status_resp.json()
                    blueprint = status.get('dna_blueprint', {})
                    
                    print(f"\n📊 DNA Blueprint Results:")
                    if isinstance(blueprint, dict) and blueprint.get('master_blueprint'):
                        master = blueprint['master_blueprint']
                        print(f"   ✅ Master blueprint found!")
                        print(f"   - Top Performers: {len(master.get('top_performers', []))}")
                        print(f"   - Content Gaps: {len(master.get('content_gaps', []))}")
                        print(f"   - Recommendations: {len(master.get('recommendations', []))}")
                    else:
                        print(f"   ⚠️  No master blueprint (may be null due to API key)")
                        
            elif result['status'] == 'already_exists':
                print(f"ℹ️  Already processed")
            else:
                print(f"❌ Failed: {result.get('error')}")
        else:
            print(f"❌ HTTP Error: {response.text}")
            
    except Exception as e:
        print(f"❌ API Error: {e}")

def main():
    print("🔍 Checking database schema and finding products...")
    print("=" * 60)
    
    # Check schema
    check_schema()
    
    # Find products with data
    products = find_products_with_data()
    
    if products:
        # Process first product with data
        product_id = products[0].get('product_id')
        process_via_api(product_id)

if __name__ == "__main__":
    main()
