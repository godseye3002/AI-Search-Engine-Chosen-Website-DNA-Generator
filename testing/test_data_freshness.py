#!/usr/bin/env python3
"""
Test the new data freshness functionality
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

def test_data_freshness():
    """Test the new data freshness check system"""
    print("🧪 Testing Data Freshness System")
    print("=" * 60)
    
    # Test product
    product_id = "02f92e70-7b53-45b6-bdef-7ef36d8fc578"
    source = "google"
    
    print(f"📦 Testing Product: {product_id}")
    print(f"📊 Data Source: {source}")
    print("-" * 40)
    
    # Step 1: Check current input data
    try:
        result = supabase.table("product_analysis_google")\
            .select("id, product_id, raw_serp_results")\
            .eq("product_id", product_id)\
            .execute()
        
        if result.data:
            input_rows = result.data
            print(f"📋 Found {len(input_rows)} input rows")
            
            # Generate hash manually to show concept
            import hashlib
            ids = [str(row['id']) for row in input_rows]
            sorted_ids = sorted(ids)
            combined_ids = ",".join(sorted_ids)
            current_hash = hashlib.sha256(combined_ids.encode('utf-8')).hexdigest()
            print(f"🔐 Current Input Hash: {current_hash[:16]}...")
            
        else:
            print("❌ No input data found")
            return
            
    except Exception as e:
        print(f"❌ Error checking input data: {e}")
        return
    
    print("\n" + "-" * 40)
    
    # Step 2: Test API processing
    print("🚀 Testing API Processing...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/process",
            json={
                "product_id": product_id,
                "source": source
            },
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.ok:
            result = response.json()
            print(f"✅ Processing Status: {result['status']}")
            print(f"📝 Message: {result.get('message', 'No message')}")
            
            if result.get('existing_analysis_id'):
                print(f"🆔 Existing Analysis ID: {result['existing_analysis_id']}")
            
            if result['status'] == 'completed':
                print(f"🎉 Full Pipeline Executed!")
                print(f"📁 Run ID: {result.get('run_id')}")
                print(f"🆔 New Analysis ID: {result.get('analysis_id')}")
                
            elif result['status'] == 'skipped':
                print(f"⏭️  Processing Skipped - Data is up to date")
                
            elif result['status'] == 'reprocessing':
                print(f"🔄 Reprocessing - Data has changed")
                
        else:
            print(f"❌ API Error: {response.text}")
            
    except Exception as e:
        print(f"❌ API Request Failed: {e}")
    
    print("\n" + "=" * 60)
    
    # Step 3: Check stored hash in database
    print("🔍 Checking Stored Hash in Database...")
    
    try:
        result = supabase.table("product_analysis_dna_google")\
            .select("product_id, input_data_hash, created_at, updated_at")\
            .eq("product_id", product_id)\
            .order("created_at", desc=True)\
            .execute()
        
        if result.data:
            # Find the record with matching hash (newest first)
            matching_record = None
            for record in result.data:
                if record.get('input_data_hash') == current_hash:
                    matching_record = record
                    break
            
            if matching_record:
                stored_hash = matching_record.get('input_data_hash')
                print(f"🔐 Stored Hash: {stored_hash[:16]}... (MATCHES)")
                print(f"📅 Created: {matching_record.get('created_at')}")
                print(f"🔄 Updated: {matching_record.get('updated_at')}")
            else:
                # Show newest record if no match found
                newest_record = result.data[0]
                stored_hash = newest_record.get('input_data_hash')
                print(f"🔐 Stored Hash: {stored_hash[:16]}... (NO MATCH)")
                print(f"📅 Created: {newest_record.get('created_at')}")
                print(f"🔄 Updated: {newest_record.get('updated_at')}")
        else:
            print("❌ No DNA analysis record found")
            
    except Exception as e:
        print(f"❌ Error checking stored hash: {e}")
    
    print("\n🎯 Data Freshness Test Complete!")
    print("=" * 60)

def test_modified_scenario():
    """Test scenario where input data is modified"""
    print("\n🧪 Testing Modified Data Scenario")
    print("=" * 60)
    
    # This would require manually modifying input data
    # For now, just show the concept
    print("💡 To test data modification:")
    print("   1. Add a new row to product_analysis_google")
    print("   2. Run this test again")
    print("   3. Should detect hash change and reprocess")
    print("   4. New hash should be stored in database")

if __name__ == "__main__":
    test_data_freshness()
    test_modified_scenario()