#!/usr/bin/env python3
"""
Force a fresh run of the complete pipeline
"""

import requests
import json
import os
from dotenv import load_dotenv
from supabase import create_client
import uuid

# Load environment
load_dotenv()

# Initialize Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def create_test_product():
    """Create a test product with real SERP data"""
    try:
        # Use the existing SERP data but with a new product ID
        existing_result = supabase.table("product_analysis_google")\
            .select("raw_serp_results, search_query")\
            .eq("product_id", "02f92e70-7b53-45b6-bdef-7ef36d8fc578")\
            .execute()
        
        if existing_result.data:
            existing = existing_result.data[0]
            
            # Create new product ID
            new_product_id = str(uuid.uuid4())
            
            # Insert with new ID
            new_product = {
                "product_id": new_product_id,
                "search_query": existing.get('search_query', {}),
                "raw_serp_results": existing.get('raw_serp_results', {}),
                "google_overview_analysis": existing.get('google_overview_analysis', {})
            }
            
            result = supabase.table("product_analysis_google").insert(new_product).execute()
            
            if result.data:
                print(f"✅ Created test product: {new_product_id}")
                
                # Show SERP data info
                raw_serp = existing.get('raw_serp_results', {})
                source_links = raw_serp.get('source_links', [])
                print(f"📝 Query: {raw_serp.get('query', 'Unknown')}")
                print(f"🔗 Source Links: {len(source_links)}")
                
                return new_product_id
            else:
                print(f"❌ Failed to create test product")
                return None
        else:
            print(f"❌ No existing product found")
            return None
            
    except Exception as e:
        print(f"❌ Error creating test product: {e}")
        return None

def run_fresh_pipeline(product_id):
    """Run the complete pipeline on a fresh product"""
    BASE_URL = "http://localhost:8000"
    
    print(f"\n🚀 Running Fresh Pipeline Test")
    print("=" * 60)
    print(f"🆔 Product ID: {product_id}")
    print("-" * 40)
    
    # Step 1: Process the product
    print("🔄 Step 1: Starting DNA Pipeline...")
    print("   This will run: Stage 1 → Stage 2 → Stage 3 → Database")
    
    import time
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
        
        processing_time = time.time() - start_time
        print(f"\n📡 Response Status: {response.status_code}")
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        
        if response.ok:
            result = response.json()
            print(f"✅ Final Status: {result['status']}")
            
            if result['status'] == 'completed':
                print(f"\n🎉 FULL PIPELINE SUCCESS!")
                print(f"📁 Run ID: {result.get('run_id')}")
                print(f"🆔 Analysis ID: {result.get('analysis_id')}")
                print(f"📂 Output Path: {result.get('final_output_path')}")
                
                # Step 2: Get the complete DNA blueprint
                print(f"\n🔍 Step 2: Retrieving Master Blueprint...")
                
                status_response = requests.post(
                    f"{BASE_URL}/status",
                    json={"product_id": product_id},
                    headers={"Content-Type": "application/json"}
                )
                
                if status_response.ok:
                    status = status_response.json()
                    blueprint = status.get('dna_blueprint', {})
                    
                    print(f"\n📊 DNA Blueprint Analysis:")
                    print(f"   📅 Created: {status.get('created_at')}")
                    
                    if isinstance(blueprint, dict):
                        if blueprint.get('query'):
                            print(f"   🔍 Query: {blueprint['query']}")
                        
                        if blueprint.get('master_blueprint'):
                            master = blueprint['master_blueprint']
                            print(f"\n   ✅ Master Blueprint Generated Successfully!")
                            print(f"   📈 Analysis Results:")
                            print(f"      - Top Performers Found: {len(master.get('top_performers', []))}")
                            print(f"      - Content Gaps Identified: {len(master.get('content_gaps', []))}")
                            print(f"      - Recommendations Generated: {len(master.get('recommendations', []))}")
                            
                            # Show detailed results
                            if master.get('top_performers'):
                                print(f"\n   🏆 Top Performers:")
                                for i, performer in enumerate(master['top_performers'][:2]):
                                    print(f"      {i+1}. {performer.get('url', 'N/A')[:60]}...")
                                    print(f"         Score: {performer.get('overall_score', 'N/A')}")
                                    print(f"         Classification: {performer.get('classification', 'N/A')}")
                            
                            if master.get('content_gaps'):
                                print(f"\n   📋 Content Gaps:")
                                for i, gap in enumerate(master['content_gaps'][:2]):
                                    print(f"      {i+1}. {gap.get('topic', 'N/A')}")
                                    print(f"         Priority: {gap.get('priority', 'N/A')}")
                            
                            if master.get('recommendations'):
                                print(f"\n   💡 Recommendations:")
                                for i, rec in enumerate(master['recommendations'][:2]):
                                    print(f"      {i+1}. {rec.get('title', 'N/A')}")
                                    print(f"         Priority: {rec.get('priority', 'N/A')}")
                                    print(f"         Impact: {rec.get('expected_impact', 'N/A')}")
                            
                            print(f"\n   🎯 Pipeline Stages Completed:")
                            print(f"      ✅ Stage 1: Website Classification")
                            print(f"      ✅ Stage 2: DNA Analysis")
                            print(f"      ✅ Stage 3: Master Blueprint Generation")
                            print(f"      ✅ Database: Results Saved")
                            
                        else:
                            print(f"   ⚠️  Master Blueprint is null")
                            print(f"   🔧 This indicates a Gemini API key issue")
                            
                    else:
                        print(f"   📄 Blueprint format: {type(blueprint)}")
                        
                else:
                    print(f"❌ Failed to get status: {status_response.status_code}")
                    
            elif result['status'] == 'skipped':
                print(f"⏭️  Product was skipped")
                print(f"📝 Reason: {result.get('message', 'Unknown')}")
                
            else:
                print(f"❌ Pipeline failed: {result.get('error')}")
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"📄 Error Details: {response.text}")
            
    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Fresh Pipeline Test Complete!")

def main():
    print("🧪 GodsEye DNA Pipeline - Fresh Run Test")
    print("This will create a new product and run the complete pipeline")
    print("=" * 60)
    
    # Create test product
    product_id = create_test_product()
    
    if product_id:
        # Run fresh pipeline
        run_fresh_pipeline(product_id)
    else:
        print("❌ Could not create test product")

if __name__ == "__main__":
    main()
