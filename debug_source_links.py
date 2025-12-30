#!/usr/bin/env python3
"""
Debug source links counting for perplexity data
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

def debug_perplexity_source_links():
    """Debug why only 5 source links found"""
    print("🔍 Debugging Perplexity Source Links")
    print("=" * 60)
    
    product_id = "02f92e70-7b53-45b6-bdef-7ef36d8fc578"
    
    try:
        # Get ALL rows for this product
        result = supabase.table("product_analysis_perplexity")\
            .select("id, product_id, raw_serp_results")\
            .eq("product_id", product_id)\
            .execute()
        
        if not result.data:
            print("❌ No data found")
            return
        
        print(f"📊 Total rows found: {len(result.data)}")
        
        total_source_links = 0
        for i, row in enumerate(result.data):
            print(f"\n--- Row {i+1} ---")
            print(f"ID: {row['id']}")
            print(f"Product ID: {row['product_id']}")
            
            # Parse raw_serp_results
            raw_serp = row.get('raw_serp_results', {})
            if isinstance(raw_serp, str):
                try:
                    raw_serp = json.loads(raw_serp)
                except:
                    raw_serp = {}
            
            # Count source links in this row
            source_links = raw_serp.get('source_links', [])
            links_count = len(source_links) if isinstance(source_links, list) else 0
            
            print(f"Source Links in this row: {links_count}")
            
            # Show some sample links
            if isinstance(source_links, list) and source_links:
                for j, link in enumerate(source_links[:3]):
                    print(f"  {j+1}. {str(link)[:80]}...")
            
            total_source_links += links_count
            
            # Show query if available (skip for perplexity as it doesn't have search_query column)
            print(f"Query: Available in raw_serp_results")
        
        print(f"\n📈 SUMMARY:")
        print(f"Total rows: {len(result.data)}")
        print(f"Total source links across all rows: {total_source_links}")
        print(f"Average links per row: {total_source_links / len(result.data):.1f}")
        
        # Check how the pipeline processes this
        print(f"\n🔧 Pipeline Processing Analysis:")
        print("The pipeline converts raw_serp_results to AI response format.")
        print("It should combine ALL source_links from ALL rows.")
        
        # Test the conversion process
        print(f"\n🧪 Testing AI Response Conversion:")
        all_source_links = []
        
        for row in result.data:
            raw_serp = row.get('raw_serp_results', {})
            if isinstance(raw_serp, str):
                try:
                    raw_serp = json.loads(raw_serp)
                except:
                    raw_serp = {}
            
            source_links = raw_serp.get('source_links', [])
            if isinstance(source_links, list):
                all_source_links.extend(source_links)
        
        print(f"Combined source links: {len(all_source_links)}")
        
        # Remove duplicates
        unique_links = []
        seen = set()
        for link in all_source_links:
            if isinstance(link, dict):
                link_str = str(link.get('url', ''))
            else:
                link_str = str(link)
            
            if link_str not in seen:
                seen.add(link_str)
                unique_links.append(link)
        
        print(f"Unique source links: {len(unique_links)}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")

def debug_google_comparison():
    """Compare with Google data for same product"""
    print(f"\n🔄 Comparing with Google Data")
    print("-" * 40)
    
    product_id = "02f92e70-7b53-45b6-bdef-7ef36d8fc578"
    
    try:
        # Get Google data
        google_result = supabase.table("product_analysis_google")\
            .select("id, product_id, raw_serp_results")\
            .eq("product_id", product_id)\
            .execute()
        
        if google_result.data:
            print(f"📊 Google rows: {len(google_result.data)}")
            
            google_total_links = 0
            for row in google_result.data:
                raw_serp = row.get('raw_serp_results', {})
                if isinstance(raw_serp, str):
                    try:
                        raw_serp = json.loads(raw_serp)
                    except:
                        raw_serp = {}
                
                source_links = raw_serp.get('source_links', [])
                google_total_links += len(source_links) if isinstance(source_links, list) else 0
            
            print(f"Google total source links: {google_total_links}")
        
    except Exception as e:
        print(f"❌ Google comparison failed: {e}")

def main():
    """Main debug function"""
    debug_perplexity_source_links()
    debug_google_comparison()

if __name__ == "__main__":
    main()
